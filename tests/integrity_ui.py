"""Streamlit privacy, consent scoping and rerun regression tests."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
import src.integrity_ui as ui


def pdf(text):
    import fitz
    doc=fitz.open()
    page=doc.new_page()
    page.insert_textbox(fitz.Rect(60,60,540,760),text,fontsize=11)
    data=doc.tobytes();doc.close()
    return data


def upload(name,data):return SimpleNamespace(name=name,getvalue=lambda:data)


class IntegrityUITests(unittest.TestCase):
    def setUp(self):
        phrase='The public authority must publish annual reports for all participating agencies'
        self.uploads=[upload('submitted.pdf',pdf(f'"{phrase}" [1].\nReferences\n[1] Smith, J. (2020). Research policy study.')),upload('reference.pdf',pdf(phrase))]
        self.file_patch=patch.object(ui.st,'file_uploader',side_effect=lambda label,**kwargs:self.uploads[0] if label=='Research paper PDF' else [self.uploads[1]])
        self.settings_patch=patch.object(ui,'detector_settings',return_value={})
        self.file_patch.start();self.settings_patch.start()
        from src.local_ai_writing import assess_local_writing
        self.local_patch=patch.object(ui,'assess_local_writing',side_effect=lambda paper,**kwargs:assess_local_writing(paper,enabled=False))
        self.local_patch.start()
        self.app=AppTest.from_string('from src.integrity_ui import render_integrity_analysis\nrender_integrity_analysis()',default_timeout=30)

    def tearDown(self):
        self.file_patch.stop();self.settings_patch.stop();self.local_patch.stop()

    def run_button(self):
        return next(b for b in self.app.button if b.label=='Run Research Paper Integrity Analysis')

    def consent(self):
        return next(c for c in self.app.checkbox if 'Crossref' in c.label)

    def test_local_results_report_and_no_network(self):
        with patch('httpx.Client',side_effect=AssertionError('Network must not be used without consent')):
            self.app.run()
            self.app.selectbox[0].set_value('R1').run()
            self.run_button().click().run()
            self.assertFalse(self.app.exception)
            result=next(iter(self.app.session_state['integrity_private']['results'].values()))
            self.assertTrue(result['analysis']['matches'][0]['cited'])
            self.assertTrue(result['analysis']['matches'][0]['quoted'])
            self.assertEqual(result['ai']['status'],'Experimental local AI-writing assessment')
            self.assertTrue(self.app.get('download_button'))
            self.app.run()
            self.assertFalse(self.app.exception)

    def test_explicit_metadata_consent_and_rerun_deduplication(self):
        resolve=Mock(side_effect=lambda ref,**kwargs:dict(ref,verification_status='Test metadata result'))
        with patch.object(ui,'resolve_reference',resolve):
            self.app.run()
            self.assertFalse(self.consent().value)
            self.run_button().click().run()
            resolve.assert_not_called()
            self.consent().check().run()
            resolve.assert_not_called()
            self.run_button().click().run()
            self.assertEqual(resolve.call_count,1)
            self.app.run()
            self.assertEqual(resolve.call_count,1)
            self.assertFalse(self.app.exception)
            self.consent().uncheck().run()
            result=next(r for r in self.app.session_state['integrity_private']['results'].values() if r['references'][0]['verification_status'].startswith('Parsed'))
            self.assertNotEqual(result['references'][0]['verification_status'],'Test metadata result')

    def test_configured_detector_requires_consent_and_displays_probabilities_separately(self):
        from src.research_integrity import writing_observations
        settings={'GPTZERO_API_KEY':'private-test-placeholder','GPTZERO_MODEL_VERSION':'test-requested-version'}
        def assessment(paper,consent,settings):
            if not consent:
                return writing_observations(paper)
            return dict(status='External detector assessment — not authorship proof',
                        explanation='Class confidence, not fraction of text written by AI.',provider='GPTZero',
                        requested_version='test-requested-version',reported_version='test-reported-version',
                        classification='MIXED',class_probabilities={'human':.2,'mixed':.7,'ai':.1},
                        confidence_category='medium',sentences=[{'text':paper.body.splitlines()[0],
                        'provider_raw_scores':{'generated_prob':.4},'meaning':'Provider sentence field; not text fraction'}],
                        documentation='https://support.gptzero.me/',characters_sent=len(paper.body))
        with patch.object(ui,'detector_settings',return_value=settings),patch.object(ui,'assess_writing',side_effect=assessment) as assess:
            self.app.run()
            next(s for s in self.app.selectbox if s.label=='AI-writing assessment method').set_value('GPTZero (optional external)').run()
            self.run_button().click().run()
            self.assertFalse(assess.call_args.kwargs['consent'])
            detector=next(c for c in self.app.checkbox if 'GPTZero' in c.label)
            detector.check().run()
            self.assertEqual(assess.call_count,1)
            self.run_button().click().run()
            self.assertTrue(assess.call_args.kwargs['consent'])
            self.assertFalse(self.app.exception)
            result=next(r for r in self.app.session_state['integrity_private']['results'].values() if r['ai'].get('provider'))
            self.assertEqual(result['ai']['class_probabilities']['mixed'],.7)
            self.assertNotIn('ai_percentage',result['ai'])
            self.assertNotIn('private-test-placeholder',str(result))

    def test_detector_highlights_metrics_report_and_rerun(self):
        from src.ai_writing import assess_writing
        settings={'GPTZERO_API_KEY':'private-test-key','GPTZERO_MODEL_VERSION':'account-version','GPTZERO_MAX_CHARACTERS':'5000'}
        client=Mock()
        def assessment(paper,consent,settings):
            client.post.return_value.json.return_value={'version':'reported-version','documents':[dict(
                document_classification='AI_ONLY',class_probabilities={'human':.1,'mixed':.2,'ai':.7},
                sentences=[{'sentence':paper.body.strip(),'highlight_sentence_for_ai':True}])]}
            return assess_writing(paper,consent,settings,client)
        with patch.object(ui,'detector_settings',return_value=settings),patch.object(ui,'assess_writing',side_effect=assessment):
            self.app.run()
            next(s for s in self.app.selectbox if s.label=='AI-writing assessment method').set_value('GPTZero (optional external)').run()
            next(c for c in self.app.checkbox if 'GPTZero' in c.label).check().run()
            client.post.assert_not_called()
            self.run_button().click().run()
            self.assertFalse(self.app.exception)
            client.post.assert_called_once()
            metrics={m.label:m.value for m in self.app.metric}
            self.assertEqual(metrics['AI-generation probability'],'70.0%')
            self.assertEqual(metrics['Percentage of analyzed text flagged as potentially AI-generated'],'100.0%')
            self.assertTrue(any('<mark>' in m.value for m in self.app.markdown))
            self.assertTrue(self.app.get('download_button'))
            self.app.run()
            client.post.assert_called_once()
            self.assertFalse(self.app.exception)

    def test_free_local_detector_does_not_require_keys_or_send_paper(self):
        from src.local_ai_writing import assess_local_writing
        from unittest.mock import Mock
        import re
        class Tokenizer:
            def num_special_tokens_to_add(self,pair=False):return 2
            def __call__(self,text,add_special_tokens=True,**kwargs):
                spans=[m.span() for m in re.finditer(r'\w+|[^\w\s]',text)]
                return dict(input_ids=[1]*(len(spans)+(2 if add_special_tokens else 0)),offset_mapping=spans)
        def assess(paper,**kwargs):return assess_local_writing(paper,resource=(Tokenizer(),None,None),**kwargs)
        with patch.object(ui,'assess_local_writing',side_effect=assess) as local,patch('src.local_ai_writing.score_passage',return_value=.9),patch.object(ui,'assess_writing') as external:
            self.app.run()
            self.run_button().click().run()
            self.assertFalse(self.app.exception)
            external.assert_not_called();local.assert_called_once()
            metrics={m.label:m.value for m in self.app.metric}
            self.assertEqual(metrics['Analyzed text flagged as potentially AI-generated'],'100.0%')
            self.assertNotIn('AI-generation probability',metrics)
            self.assertTrue(any('<mark>' in m.value for m in self.app.markdown))
            self.app.run();local.assert_called_once()
            self.assertTrue(self.app.get('download_button'))

    def test_new_upload_requires_new_consent_and_clear_removes_state(self):
        self.app.run()
        self.consent().check().run()
        self.uploads[0]=upload('new.pdf',pdf('A different new paper body with sufficient readable research words and sentences.'))
        self.app.run()
        self.assertFalse(self.consent().value)
        self.assertEqual(self.app.session_state['integrity_private']['paper'].name,'new.pdf')
        self.uploads[0]=None
        # The fake uploader remains empty after clearing just as the real widget does.
        next(b for b in self.app.button if 'Clear private' in b.label).click().run()
        self.assertNotIn('integrity_private',self.app.session_state)
        self.assertFalse(self.app.exception)


if __name__=='__main__':unittest.main()
