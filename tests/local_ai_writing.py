"""Offline token budgeting, evidence, coverage and failure checks (mock scores)."""
import sys,re,threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
from src.local_ai_writing import split_passages,assess_local_writing,MODEL
from src.research_integrity import paper_from_pages,integrity_report,compare_sources

class Tokenizer:
    def num_special_tokens_to_add(self,pair=False):return 2
    def __call__(self,text,add_special_tokens=True,**kwargs):
        spans=[(m.start(),m.end()) for m in re.finditer(r'\w+|[^\w\s]',text)]
        return dict(input_ids=list(range(len(spans)+(2 if add_special_tokens else 0))),offset_mapping=spans)

def paper(pages):return paper_from_pages('fixture.pdf',[dict(page_number=i+1,text=p) for i,p in enumerate(pages)])
RESOURCE=(Tokenizer(),None,threading.Lock())

class LocalDetectorTests(unittest.TestCase):
    def test_token_splitting_no_truncation_and_every_word_covered(self):
        p=paper([' '.join('word'+str(i) for i in range(1400))])
        chunks,skipped=split_passages(p.body,Tokenizer())
        self.assertEqual(skipped,[]);self.assertEqual(len(chunks),3)
        self.assertTrue(all(c['model_tokens']<=512 for c in chunks))
        with patch('src.local_ai_writing.score_passage',side_effect=[.9,.1,.9]):
            result=assess_local_writing(p,resource=RESOURCE)
        self.assertEqual(result['analyzed_words'],1400)
        self.assertEqual(result['flagged_words'],890)
        self.assertAlmostEqual(result['flagged_percentage'],100*890/1400)
        self.assertEqual(result['coverage']['percent'],100)
        self.assertNotIn('document_probability',result)
        self.assertIn('Uncertain',result['assessment'])

    def test_bibliography_exclusion_page_attribution_and_report(self):
        p=paper(['First page claim.','Second page claim.\nReferences\n[1] Smith, J. (2020). Source study.'])
        with patch('src.local_ai_writing.score_passage',return_value=.8):result=assess_local_writing(p,resource=RESOURCE)
        self.assertEqual(result['sentences'][0]['pages'],[1,2])
        self.assertNotIn('Smith',result['sentences'][0]['text'])
        self.assertEqual(result['flagged_percentage'],100)
        report=integrity_report(p,[],compare_sources(p,[],[]),result)
        self.assertIn('Analyzed text flagged as potentially AI-generated',report)
        self.assertIn('uncalibrated',report)
        self.assertIn('not the actual percentage of AI used',report)
        self.assertIn(MODEL,report)

    def test_word_union_with_overlapping_chunks(self):
        p=paper(['alpha beta gamma'])
        chunks=[dict(text='alpha beta gamma',start=0,end=16,model_tokens=5),dict(text='beta gamma',start=6,end=16,model_tokens=4)]
        with patch('src.local_ai_writing.split_passages',return_value=(chunks,[])),patch('src.local_ai_writing.score_passage',return_value=.9):result=assess_local_writing(p,resource=RESOURCE)
        self.assertEqual(result['analyzed_words'],3);self.assertEqual(result['flagged_words'],3)
        self.assertEqual(result['flagged_percentage'],100)

    def test_passage_budget_partial_coverage(self):
        p=paper([' '.join('w'+str(i) for i in range(1200))])
        with patch('src.local_ai_writing.score_passage',return_value=.9):result=assess_local_writing(p,max_passages=1,resource=RESOURCE)
        self.assertEqual(result['analyzed_words'],510)
        self.assertEqual(result['coverage']['skipped_words'],690)
        self.assertIn('partial coverage',result['assessment'])
        self.assertTrue(result['skipped'])

    def test_model_language_memory_and_download_failures(self):
        p=paper(['actual readable text'])
        for language,enabled in [('Other / unknown',True),('English',False)]:
            with patch('src.local_ai_writing.load_detector') as loader:
                result=assess_local_writing(p,language=language,enabled=enabled);loader.assert_not_called()
                self.assertIsNone(result['flagged_percentage'])
        for error,message in [(MemoryError(),'memory'),(RuntimeError('out of memory'),'memory'),(OSError(),'download'),(ImportError(),'dependency'),(ValueError(),'compatibility')]:
            with patch('src.local_ai_writing.load_detector',side_effect=error):result=assess_local_writing(p)
            self.assertIn(message,result['error']);self.assertEqual(result['analyzed_words'],0)
            self.assertIsNone(result['flagged_percentage'])

    def test_inference_failure_is_skipped_not_fake_zero(self):
        p=paper(['readable words'])
        with patch('src.local_ai_writing.score_passage',side_effect=ValueError()):result=assess_local_writing(p,resource=RESOURCE)
        self.assertIsNone(result['flagged_percentage']);self.assertEqual(result['coverage']['skipped_words'],2)
        self.assertNotIn('observations',result)

    def test_invalid_model_output_has_no_score(self):
        for score in [float('nan'),float('inf'),-1,True]:
            with patch('src.local_ai_writing.score_passage',return_value=score):result=assess_local_writing(paper(['Readable words']),resource=RESOURCE)
            self.assertIsNone(result['flagged_percentage']);self.assertEqual(result['analyzed_words'],0)

    def test_whole_word_boundary_with_subword_tokenizer(self):
        class Characters(Tokenizer):
            def __call__(self,text,add_special_tokens=True,**kwargs):
                spans=[(i,i+1) for i,char in enumerate(text) if not char.isspace()]
                return dict(input_ids=[1]*(len(spans)+(2 if add_special_tokens else 0)),offset_mapping=spans)
        chunks,skipped=split_passages('abc defgh ijk',Characters(),limit=10)
        self.assertEqual([c['text'] for c in chunks],['abc defgh','ijk'])
        self.assertEqual(skipped,[])
        chunks,skipped=split_passages('oversizedword end',Characters(),limit=5)
        self.assertTrue(skipped);self.assertEqual(chunks[0]['text'],'end')

    def test_configuration_defaults_are_optional_and_invalid_disclosed(self):
        import os
        from src.local_ai_writing import local_defaults
        with patch('dotenv.load_dotenv'),patch.dict(os.environ,{'LOCAL_AI_THRESHOLD':'nan','LOCAL_AI_MAX_PASSAGES':'0'},clear=True):defaults=local_defaults()
        self.assertEqual(defaults['threshold'],.8);self.assertEqual(defaults['max_passages'],200)
        self.assertEqual(len(defaults['warnings']),2)

    def test_invalid_threshold_empty_input_and_scores(self):
        for threshold in [float('nan'),True,-.1,2]:
            result=assess_local_writing(paper(['words']),threshold=threshold,resource=RESOURCE)
            self.assertIn('threshold',result['error']);self.assertIsNone(result['flagged_percentage'])
        self.assertIsNone(assess_local_writing(paper(['']),resource=RESOURCE)['flagged_percentage'])

if __name__=='__main__':unittest.main()
