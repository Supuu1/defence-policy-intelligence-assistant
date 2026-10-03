"""Local evidence, citation and privacy tests; no live keys or external documents."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import httpx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import (
    paper_from_pages,citation_analysis,compare_sources,overlap_percentage,extract_pdf,
    integrity_report,writing_observations,safe_link,
)
from src.scholarly_sources import resolve_reference,retrieve_open_access,CROSSREF,EUROPE_PMC
from src.ai_writing import assess_writing


def paper(body,name='paper.pdf',page=1):
    return paper_from_pages(name,[dict(page_number=page,text=body,extraction_method='Native text')])


def source(other,**kwargs):
    return dict(paper=other,name=other.name,status='Readable',**kwargs)


class IntegrityTests(unittest.TestCase):
    def test_numbered_citations_ranges_and_missing_references(self):
        p=paper('Supported claims [1, 3–4].\nReferences\n[1] Smith, J. (2020). Study of policy.\n[3] Jones, P. (2021). Second study.')
        citations=citation_analysis(p)
        self.assertEqual([c['text'] for c in citations],['[1]','[3]','[4]'])
        self.assertEqual([c['status'] for c in citations],['Matched bibliography entry','Matched bibliography entry','Missing bibliography reference'])
        self.assertEqual(citations[0]['reference_ids'],['R1'])
        self.assertTrue(p.bibliography[1]['cited'])
        self.assertNotIn('Smith',p.body)

    def test_author_year_and_ambiguous_reference(self):
        p=paper('Smith (2020) reports findings (Jones et al., 2021). In 2022 there was a review.\nReferences\nSmith, J. (2020). Research results.\nJones, P. (2021). More results.')
        c=citation_analysis(p)
        self.assertEqual(len(c),2)
        self.assertTrue(all(x['reference_ids'] for x in c))
        p=paper('As reported (Smith, 2020).\nReferences\nSmith, J. (2020). First work.\nSmith, K. (2020). Another work.')
        c=citation_analysis(p)
        self.assertEqual(c[0]['status'],'Ambiguous bibliography match')
        self.assertFalse(any(ref['cited'] for ref in p.bibliography))
        self.assertTrue(all(ref['ambiguous_citation'] for ref in p.bibliography))

    def test_bibliography_page_attribution_and_appendix(self):
        p=paper_from_pages('multi.pdf',[
            dict(page_number=1,text='This is a claim [2].'),
            dict(page_number=2,text='References\n[2] Doe, A. (2019). A test reference.'),
            dict(page_number=3,text='Appendix A\nAdditional actual source text here.')])
        self.assertEqual(p.bibliography[0]['pages'],[2])
        self.assertEqual(citation_analysis(p)[0]['pages'],[1])
        self.assertIn('Additional actual',p.body)
        self.assertNotIn('A test reference',p.body)

    def test_quoted_and_cited_overlap(self):
        phrase='The public authority must publish annual reports for all participating agencies'
        p=paper(f'“{phrase}” [1].\nReferences\n[1] Smith, J. (2020). Source study.')
        c=citation_analysis(p)
        result=compare_sources(p,[source(paper(phrase,'reference.pdf',5),reference_ids=['R1'])],c)
        m=result['matches'][0]
        self.assertTrue(m['quoted']);self.assertTrue(m['cited'])
        self.assertEqual(m['source_pages'],[5])
        self.assertEqual(m['submitted_pages'],[1])
        self.assertEqual(result['unattributed_overlap_percentage'],0)
        self.assertGreater(result['overlap_percentage'],0)

    def test_quotation_without_source_citation_still_requires_review(self):
        phrase='The public authority must publish annual reports for all participating agencies'
        p=paper(f'“{phrase}”')
        result=compare_sources(p,[source(paper(phrase,'reference.pdf'))],[])
        self.assertTrue(result['matches'][0]['quoted'])
        self.assertFalse(result['matches'][0]['cited'])
        self.assertEqual(result['unattributed_overlap_percentage'],100)
        self.assertEqual(result['quoted_cited_overlap_percentage'],0)

    def test_topic_or_unrelated_citation_does_not_establish_use(self):
        phrase='The public authority must publish annual reports for all participating agencies'
        p=paper(phrase+' [1].\nReferences\n[1] Smith, J. (2020). A different source.')
        result=compare_sources(p,[source(paper(phrase,'uncited.pdf'))],citation_analysis(p))
        self.assertFalse(result['matches'][0]['cited'])
        self.assertIn('Potentially unattributed',result['matches'][0]['attribution'])

    def test_duplicate_matches_percentage_and_denominator(self):
        p=paper('alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu')
        s=paper('alpha beta gamma delta epsilon zeta eta theta','one.pdf')
        result=compare_sources(p,[source(s),source(s)],[])
        self.assertEqual(result['analyzed_body_words'],12)
        self.assertEqual(result['overlapping_words'],8)
        self.assertAlmostEqual(result['overlap_percentage'],100*8/12)
        self.assertEqual(len(result['matches']),1)
        self.assertIn('duplicate',result['coverage'][1]['status'])
        self.assertAlmostEqual(overlap_percentage([(0,8),(2,10),(5,8),(0,8)],12),100*10/12)

    def test_overlapping_matches_from_distinct_sources_deduplicate_word_positions(self):
        p=paper('alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu')
        one=paper('alpha beta gamma delta epsilon zeta eta theta','one.pdf')
        two=paper('epsilon zeta eta theta iota kappa lambda mu','two.pdf')
        result=compare_sources(p,[source(one),source(two)],[])
        self.assertEqual(result['overlapping_words'],12)
        self.assertEqual(result['overlap_percentage'],100)
        self.assertEqual(len(result['matches']),2)

    def test_passage_crossing_page_boundary_has_both_page_references(self):
        p=paper_from_pages('two.pdf',[dict(page_number=1,text='alpha beta gamma delta'),dict(page_number=2,text='epsilon zeta eta theta')])
        s=paper('alpha beta gamma delta epsilon zeta eta theta','reference.pdf',9)
        result=compare_sources(p,[source(s)],[])
        self.assertEqual(result['matches'][0]['submitted_pages'],[1,2])
        self.assertEqual(result['matches'][0]['source_pages'],[9])

    def test_failed_semantic_model_does_not_discard_lexical_results(self):
        p=paper('alpha beta gamma delta epsilon zeta eta theta unique body words')
        s=paper('alpha beta gamma delta epsilon zeta eta theta','reference.pdf')
        model=Mock();model.encode.side_effect=RuntimeError('private model details')
        result=compare_sources(p,[source(s)],[],model)
        self.assertGreater(result['overlap_percentage'],0)
        self.assertIn('encoding failed',result['semantic_status'])
        self.assertNotIn('private model details',str(result))

    def test_bibliography_not_counted_in_overlap(self):
        p=paper('Unique body words without source overlap.\nReferences\n[1] The public authority must publish annual reports for all participating agencies.')
        s=paper('The public authority must publish annual reports for all participating agencies.','reference.pdf')
        result=compare_sources(p,[source(s)],citation_analysis(p))
        self.assertEqual(result['overlap_percentage'],0)
        self.assertEqual(result['analyzed_body_words'],6)

    def test_near_exact_counts_matching_tokens_only(self):
        p=paper('alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu')
        s=paper('alpha beta gamma delta CHANGED zeta eta theta iota kappa lambda mu','reference.pdf')
        result=compare_sources(p,[source(s)],[])
        self.assertEqual(result['matches'][0]['match_type'],'near-exact overlap')
        self.assertEqual(result['overlapping_words'],11)
        self.assertAlmostEqual(result['overlap_percentage'],100*11/12)

    def test_unavailable_source_and_self_comparison(self):
        p=paper('This is actual readable submitted text with enough words to analyze.')
        result=compare_sources(p,[dict(name='offline source',paper=None,status='Unavailable: timeout'),source(p)],[])
        self.assertEqual(result['coverage'][0]['status'],'Unavailable: timeout')
        self.assertIn('submitted paper itself',result['coverage'][1]['status'])
        self.assertEqual(result['overlap_percentage'],0)
        self.assertIn('Zero overlap does not imply originality',result['limitations'])

    def test_partial_coverage_and_missing_bibliography_heading(self):
        p=paper(' '.join('word'+str(i) for i in range(30)))
        with patch('src.research_integrity.MAX_SUBMITTED_WORDS',10):
            result=compare_sources(p,[],[])
        self.assertEqual(result['analyzed_body_words'],10)
        self.assertEqual(result['total_body_words'],30)
        self.assertFalse(p.bibliography_found)
        self.assertTrue(p.extraction_notes)

    def test_semantic_similarity_is_separate_from_percentage(self):
        import numpy as np
        p=paper('We investigate public transport policy and its consequences for local resident travel habits.')
        s=paper('Municipal mobility decisions affect how citizens journey through their city each day.','reference.pdf')
        model=Mock()
        model.encode.side_effect=[np.array([[1.,0.]]),np.array([[1.,0.]])]
        result=compare_sources(p,[source(s)],[],model)
        self.assertEqual(result['matches'][0]['match_type'],'semantic similarity')
        self.assertEqual(result['overlap_percentage'],0)
        self.assertEqual(result['overlapping_words'],0)

    def test_no_consent_means_no_network_even_if_detector_configured(self):
        client=Mock()
        ref=paper('References\n[1] Smith, J. (2020). Test work. doi:10.5555/test').bibliography[0]
        with self.assertRaises(PermissionError): resolve_reference(ref,client=client)
        with self.assertRaises(PermissionError): retrieve_open_access(ref,client=client)
        result=assess_writing(paper('Actual paper body text.'),False,{'GPTZERO_API_KEY':'placeholder','GPTZERO_MODEL_VERSION':'test-version'},client)
        self.assertEqual(result['status'],'AI authorship assessment unavailable')
        self.assertFalse(client.mock_calls)

    def test_crossref_doi_verified_but_title_search_only_candidate(self):
        def handler(request):
            metadata={'DOI':'10.5555/test','title':['A verified record'],'author':[{'family':'Smith','given':'J'}],'published':{'date-parts':[[2020]]}}
            return httpx.Response(200,json={'message':metadata if '/works/' in request.url.path else {'items':[metadata]}})
        ref=paper('References\n[1] Smith, J. (2020). A stated title. doi:10.5555/test').bibliography[0]
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            resolved=resolve_reference(ref,True,client)
            self.assertIn('DOI record verified',resolved['verification_status'])
            self.assertEqual(resolved['title'],'A verified record')
            no_doi=dict(ref,doi='')
            candidate=resolve_reference(no_doi,True,client)
            self.assertIn('Candidate metadata only',candidate['verification_status'])
            self.assertEqual(candidate['title'],ref['title'])
            self.assertEqual(candidate['metadata_candidates'][0]['title'],'A verified record')

    def test_open_access_only_and_no_invented_page_numbers(self):
        ref=dict(id='R1',title='Verified source',doi='10.5555/test',link='',verification_status='DOI record verified in Crossref')
        def handler(request):
            if request.url.path.endswith('/search'):
                return httpx.Response(200,json={'resultList':{'result':[{'doi':'10.5555/test','pmcid':'PMC123','isOpenAccess':'Y'}]}})
            return httpx.Response(200,text='<article><body><sec><p>Actual legally accessible article text with enough words for comparison.</p></sec></body><back><ref-list><ref>Bibliography text excluded</ref></ref-list></back></article>')
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result=retrieve_open_access(ref,True,client)
        self.assertIsNotNone(result['paper'])
        self.assertIsNone(result['paper'].pages[0]['page_number'])
        self.assertNotIn('Bibliography text',result['paper'].body)
        self.assertEqual(result['reference_ids'],['R1'])
        def unavailable(request):
            return httpx.Response(200,json={'resultList':{'result':[{'doi':'10.5555/test','pmcid':'PMC123','isOpenAccess':'N'}]}})
        with httpx.Client(transport=httpx.MockTransport(unavailable)) as client:
            result=retrieve_open_access(ref,True,client)
        self.assertIsNone(result['paper']);self.assertIn('no Europe PMC open-access',result['status'])

    def test_external_errors_sanitized_and_xml_entities_rejected(self):
        ref=dict(id='R1',raw='Some reference',doi='10.5555/test',title='Source',link='',verification_status='DOI record verified')
        def quota(request):return httpx.Response(429,text='PRIVATE_CONTENT_AND_HEADERS')
        with httpx.Client(transport=httpx.MockTransport(quota)) as client:
            result=resolve_reference(ref,True,client)
        self.assertIn('rate limit',result['verification_status'])
        self.assertNotIn('PRIVATE_CONTENT',str(result))
        def entity(request):
            if request.url.path.endswith('/search'):
                return httpx.Response(200,json={'resultList':{'result':[{'doi':'10.5555/test','pmcid':'PMC123','isOpenAccess':'Y'}]}})
            return httpx.Response(200,text='<!DOCTYPE article [<!ENTITY x SYSTEM "file:///etc/passwd">]><article><body><p>&x;</p></body></article>')
        with httpx.Client(transport=httpx.MockTransport(entity)) as client:
            result=retrieve_open_access(ref,True,client)
        self.assertIsNone(result['paper'])

    def test_detector_scores_keep_provider_meaning(self):
        p=paper('Actual paper sentence with sufficiently readable text.')
        client=Mock()
        client.post.return_value.json.return_value={'version':'reported-model','documents':[dict(document_classification='MIXED',class_probabilities={'human':.2,'mixed':.7,'ai':.1},sentences=[{'sentence':p.body.strip(),'generated_prob':.4}])]}
        result=assess_writing(p,True,{'GPTZERO_API_KEY':'private-placeholder','GPTZERO_MODEL_VERSION':'requested-model','GPTZERO_MAX_CHARACTERS':'5000'},client)
        self.assertEqual(result['class_probabilities']['mixed'],.7)
        self.assertEqual(result['reported_version'],'reported-model')
        self.assertNotIn('ai_percentage',result)
        self.assertIn('not the proportion',result['explanation'])
        self.assertNotIn('private-placeholder',str(result))

    def test_detector_malformed_response_does_not_infer_score(self):
        client=Mock();client.post.return_value.json.return_value={'documents':[{'document_classification':'MIXED','class_probabilities':{'ai':120}}]}
        result=assess_writing(paper('Actual text.'),True,{'GPTZERO_API_KEY':'placeholder','GPTZERO_MODEL_VERSION':'test','GPTZERO_MAX_CHARACTERS':'5000'},client)
        self.assertEqual(result['status'],'AI authorship assessment unavailable')
        self.assertNotIn('class_probabilities',result)

    def test_report_evidence_coverage_and_no_misconduct_claim(self):
        p=paper('alpha beta gamma delta epsilon zeta eta theta extra words here')
        s=paper('alpha beta gamma delta epsilon zeta eta theta','reference.pdf')
        result=compare_sources(p,[source(s,link='https://example.org/source')],[])
        report=integrity_report(p,[],result,writing_observations(p))
        self.assertIn('Similarity within checked sources',report)
        self.assertIn('denominator (examined body tokens): 11',report)
        self.assertIn('alpha beta gamma delta epsilon zeta eta theta',report)
        self.assertIn('https://example.org/source',report)
        self.assertIn('does not imply originality',report)
        self.assertIn('AI authorship assessment unavailable',report)
        self.assertEqual(safe_link('javascript:alert(1)'), '')
        self.assertEqual(safe_link('https://user:password@example.com'),'')

    def test_pdf_extraction_in_memory_and_pages(self):
        import fitz
        pdf=fitz.open()
        for i in range(2):
            pdf.new_page().insert_text((72,72),f'This is actual readable paper text on original page {i+1} with enough words.')
        data=pdf.tobytes();pdf.close()
        p=extract_pdf(data,'test.pdf')
        self.assertEqual([x['page_number'] for x in p.pages],[1,2])
        self.assertIn('original page 2',p.body)


if __name__=='__main__': unittest.main()
