"""Detector evidence, privacy and percentage regression checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import Mock, patch
import httpx
from src.ai_writing import assess_writing, highlighted_passage, passage_results, detector_settings
from src.research_integrity import paper_from_pages, integrity_report, compare_sources

SETTINGS={'GPTZERO_API_KEY':'private-key','GPTZERO_MODEL_VERSION':'account-test-version','GPTZERO_MAX_CHARACTERS':'5000'}

def paper(text):
    return paper_from_pages('paper.pdf',[{'text':text,'page_number':1}],digest='test')

def response(sentences,classification='AI_ONLY'):
    return {'version':'provider-reported-version','documents':[{'document_classification':classification,
        'class_probabilities':{'human':.1,'mixed':.2,'ai':.7},'confidence_category':'medium','sentences':sentences}]}

class DetectorTests(unittest.TestCase):
    def run_detector(self,p,data,settings=SETTINGS):
        client=Mock(); client.post.return_value.json.return_value=data
        return assess_writing(p,True,settings,client),client

    def test_genuine_flags_percentage_dedup_and_pages(self):
        p=paper_from_pages('paper.pdf',[{'text':'Alpha beta.','page_number':1},{'text':'Gamma delta.\nReferences\n[1] Reference words.','page_number':2}],digest='test')
        rows=[{'sentence':'Alpha beta.','highlight_sentence_for_ai':True}]*2+[{'sentence':'Gamma delta.','highlight_sentence_for_ai':False}]
        result,client=self.run_detector(p,response(rows))
        self.assertEqual(result['flagged_percentage'],50)
        self.assertEqual(result['flagged_words'],2)
        self.assertEqual(result['analyzed_words'],4)
        self.assertEqual(result['sentences'][1]['pages'],[2])
        self.assertEqual(result['ai_generation_probability'],.7)
        payload=client.post.call_args.kwargs['json']
        self.assertNotIn('Reference words',payload['document'])
        self.assertEqual(payload['version'],'account-test-version')
        self.assertNotIn('private-key',str(result))
        report=integrity_report(p,[],compare_sources(p,[],[]),result)
        self.assertIn('50.0%',report)
        self.assertIn('provider-reported-version',report)

    def test_no_flags_no_percentage_or_invented_highlights(self):
        p=paper('Alpha beta.')
        result,_=self.run_detector(p,response([{'sentence':'Alpha beta.','generated_prob':.99}]))
        self.assertIsNone(result['flagged_percentage'])
        self.assertIsNone(result['sentences'][0]['flagged'])

    def test_incomplete_passages_do_not_imply_zero(self):
        result,_=self.run_detector(paper('Alpha beta. Gamma delta.'),response([{'sentence':'Alpha beta.','highlight_sentence_for_ai':False}]))
        self.assertIsNone(result['flagged_percentage'])
        self.assertEqual(result['passage_covered_words'],2)

    def test_ambiguous_repeated_and_foreign_sentences_not_inferred(self):
        p=paper('Repeat words. Repeat words.')
        result=passage_results(p,p.body,[{'sentence':'Repeat words.','highlight_sentence_for_ai':True},{'sentence':'Invented sentence.','highlight_sentence_for_ai':True}])
        self.assertEqual(result['sentences'],[])
        self.assertIsNone(result['flagged_percentage'])

    def test_overlapping_flags_union_word_positions(self):
        p=paper('Alpha beta gamma.')
        result=passage_results(p,p.body,[{'sentence':'Alpha beta gamma.','highlight_sentence_for_ai':True},{'sentence':'beta gamma.','highlight_sentence_for_ai':True}])
        self.assertEqual(result['flagged_words'],3)
        self.assertEqual(result['flagged_percentage'],100)

    def test_long_paper_partial_no_chunk_aggregation(self):
        p=paper('Alpha beta gamma. Delta epsilon zeta.')
        result,client=self.run_detector(p,response([]),dict(SETTINGS,GPTZERO_MAX_CHARACTERS='20'))
        client.post.assert_called_once()
        self.assertLessEqual(len(client.post.call_args.kwargs['json']['document']),20)
        self.assertTrue(result['coverage']['partial'])
        self.assertLess(result['coverage']['percent'],100)
        self.assertEqual(result['ai_generation_probability'],.7)

    def test_explicit_consent_credentials_configuration_empty_input(self):
        for consent,settings,p in [(False,SETTINGS,paper('Alpha beta.')),(True,{},paper('Alpha beta.')),
            (True,dict(SETTINGS,GPTZERO_MAX_CHARACTERS='invalid'),paper('Alpha beta.')),
            (True,SETTINGS,paper(''))]:
            client=Mock(); result=assess_writing(p,consent,settings,client)
            client.post.assert_not_called()
            self.assertNotIn('provider',result)

    def test_error_categories_sanitized_and_no_retries(self):
        for status,expected in [(401,'Authentication'),(403,'Authentication'),(402,'Quota'),(429,'Quota'),(422,'configuration'),(413,'configuration'),(500,'service')]:
            request=httpx.Request('POST','https://api.gptzero.me/v2/predict/text')
            r=httpx.Response(status,request=request,text='private-key SECRET_PAPER')
            client=Mock();client.post.side_effect=httpx.HTTPStatusError('private-key SECRET_PAPER',request=request,response=r)
            result=assess_writing(paper('Alpha beta.'),True,SETTINGS,client)
            self.assertIn(expected,result['explanation'])
            self.assertNotIn('private-key',str(result)); self.assertNotIn('SECRET_PAPER',str(result))
            client.post.assert_called_once()
        for error,expected in [(httpx.ReadTimeout('SECRET_PAPER'),'timed out'),(httpx.ConnectError('SECRET_PAPER'),'connectivity')]:
            client=Mock();client.post.side_effect=error
            result=assess_writing(paper('Alpha beta.'),True,SETTINGS,client)
            self.assertIn(expected,result['explanation'])
            self.assertNotIn('SECRET_PAPER',str(result))

    def test_empty_malformed_response_unavailable(self):
        for data in [{},{'documents':[]},response([],classification='INVENTED'),response([None])]:
            result,_=self.run_detector(paper('Alpha beta.'),data)
            if 'provider' in result:
                self.assertIsNone(result['flagged_percentage'])
            else:
                self.assertIn('Malformed',result['explanation'])

    def test_credentials_environment_then_secrets_without_logging(self):
        import streamlit as st
        import os
        secret_settings=dict(SETTINGS,GPTZERO_API_KEY='secret-key')
        with patch('dotenv.load_dotenv'),patch.dict(os.environ,{'GPTZERO_API_KEY':'env-key'},clear=True),patch.object(st,'secrets',secret_settings):
            result=detector_settings()
        self.assertEqual(result['GPTZERO_API_KEY'],'env-key')
        self.assertEqual(result['GPTZERO_MODEL_VERSION'],'account-test-version')
        self.assertEqual(result['GPTZERO_MAX_CHARACTERS'],'5000')

    def test_html_escape(self):
        result=highlighted_passage({'text':'<script>alert("key")</script>','flagged':True})
        self.assertIn('<mark>',result);self.assertNotIn('<script>',result)

if __name__=='__main__':unittest.main()
