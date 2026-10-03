"""Structural reference regressions, with no private document fixtures."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import Mock,patch
import httpx
from src.research_integrity import paper_from_pages,citation_analysis,integrity_report,compare_sources,writing_observations
from src.scholarly_sources import resolve_reference
from src.report_service import markdown_table

def paper(*pages):return paper_from_pages('fixture.pdf',[dict(page_number=i+1,text=p) for i,p in enumerate(pages)])

class BibliographyTests(unittest.TestCase):
    def test_years_and_mentions_do_not_start_bibliography(self):
        p=paper('Introduction\nIn 1998, Steane described classical information theory.\nThe references are discussed below.\nClassical information theory\nA body paragraph mentions Smith (2020).')
        self.assertFalse(p.bibliography_found);self.assertEqual(p.bibliography,[])
        self.assertIn('Classical information theory',p.body)

    def test_toc_and_year_paragraph_after_heading_are_rejected(self):
        p=paper('Contents\nReferences .... 16\nIntroduction\nReferences\nThe study published in 2020 is described in the following body paragraph.\nConclusion')
        self.assertFalse(p.bibliography_found);self.assertEqual(p.bibliography,[])

    def test_real_section_multiline_across_pages_headers_removed(self):
        p=paper('Quantum Computing Review\nBody claim [1].\nMore body text.\nReferences\n[1] Smith, J. (2020). A long\n1',
                'Quantum Computing Review\nreference title continued.\nJournal 10, pages 12–18.\n[2] Jones, P. (2021). Another work.\n2')
        self.assertEqual(len(p.bibliography),2)
        self.assertEqual(p.bibliography[0]['pages'],[1,2])
        self.assertIn('reference title continued',p.bibliography[0]['title'])
        self.assertNotIn('Quantum Computing Review',p.bibliography[0]['raw'])
        self.assertEqual(citation_analysis(p)[0]['reference_ids'],['R1'])

    def test_numbered_and_author_year_no_invented_fields(self):
        p=paper('As shown (Smith, 2020) [4].\nBibliography\nSmith, J. (2020). A real study.\n[4] Unclear entry with no validated author or year')
        self.assertEqual(p.bibliography[1]['parse_status'],'Unparsed reference')
        self.assertEqual(p.bibliography[1]['authors'],'');self.assertEqual(p.bibliography[1]['year'],'')
        self.assertTrue(all(c['reference_ids'] for c in citation_analysis(p)))
        client=Mock();r=resolve_reference(p.bibliography[1],True,client)
        client.stream.assert_not_called();self.assertIn('not submitted',r['verification_status'])

    def test_initial_first_physics_reference_and_trailing_year(self):
        p=paper('Claim [1].\nReferences\n[1] A. M. Steane, “Quantum computing”, Reports on Progress in Physics 61, 117 (1998).')
        self.assertEqual(p.bibliography[0]['surname'],'steane')
        self.assertEqual(p.bibliography[0]['title'],'Quantum computing')
        self.assertEqual(p.bibliography[0]['year'],'1998')

    def test_short_markdown_cells_escape_pipes_newlines_and_evidence_outside(self):
        table=markdown_table(['Authors','Title'],[['Smith | Jones','Very long title '*40+'\nline']])
        self.assertIn('Smith \\| Jones',table);self.assertEqual(len(table.splitlines()),3)
        self.assertLess(len(table.splitlines()[2]),140)
        p=paper('Claims [1].\nReferences\n[1] Smith, J. (2020). A title | with delimiter. Journal evidence.')
        report=integrity_report(p,citation_analysis(p),compare_sources(p,[],[]),writing_observations(p))
        self.assertIn('A title \\| with delimiter',report)
        self.assertIn('Bibliography evidence (outside the compact table)',report)

    def test_rate_limit_retry_after_and_private_success_cache(self):
        ref=paper('References\n[1] Smith, J. (2020). Real study. doi:10.5555/test').bibliography[0]
        calls=[]
        def handler(request):
            calls.append(request)
            if len(calls)==1:return httpx.Response(429,headers={'Retry-After':'2'})
            return httpx.Response(200,json={'message':{'DOI':'10.5555/test','title':['Real study']}})
        cache={}
        with httpx.Client(transport=httpx.MockTransport(handler)) as client,patch('src.scholarly_sources.time.sleep') as sleep:
            result=resolve_reference(ref,True,client,cache);again=resolve_reference(ref,True,client,cache)
        self.assertEqual(len(calls),2);self.assertGreaterEqual(sleep.call_args.args[0],2)
        self.assertTrue(again['metadata_cache_hit']);self.assertIn('verified',result['verification_status'])

    def test_headingless_reference_run_is_structural_and_late(self):
        body='Introduction\n'+('Ordinary body paragraph mentions years 1998 and 2020.\n'*30)
        refs='\n'.join(f'Surname{i} A 2020 A valid study number {i}, Phys. Rev. 10, 12.' for i in range(8))
        p=paper(body,refs)
        self.assertTrue(p.bibliography_found);self.assertEqual(len(p.bibliography),8)
        self.assertTrue(all(r['lookup_eligible'] for r in p.bibliography))
        self.assertIn('Introduction',p.body)

    def test_prose_cannot_be_an_author_list_or_reference_continuation(self):
        p=paper('References\n[1] Smith, J. This is an entire body paragraph before the year (2020). Alleged title.\n[2] Jones, A. (2021). Real title.\nThe '+('ordinary body words '*20)+'.')
        self.assertEqual(p.bibliography[0]['authors'],'')
        self.assertFalse(p.bibliography[0]['lookup_eligible'])
        self.assertTrue(p.bibliography[1]['lookup_eligible'])
        self.assertNotIn('ordinary body',p.bibliography[1]['raw'])

    def test_reference_title_starting_introduction_is_not_section_heading(self):
        p=paper('Claim [1].\nReferences\n[1] Smith, J. (2020).\nIntroduction to quantum mechanics. Publisher.\n[2] Jones, A. (2021). Another title.')
        self.assertEqual(len(p.bibliography),2)
        self.assertEqual(p.bibliography[0]['title'],'Introduction to quantum mechanics')

    def test_standalone_reference_year_is_not_removed_as_page_footer(self):
        p=paper('References\n[1] Smith, J.\nA title with no inferred year.\n2020')
        self.assertIn('2020',p.bibliography[0]['raw'])

    def test_retry_bound_and_excessive_retry_after(self):
        ref=paper('References\n[1] Smith, J. (2020). Real study.').bibliography[0]
        for header,count in [({},3),({'Retry-After':'120'},1)]:
            calls=[]
            def handler(request):calls.append(request);return httpx.Response(429,headers=header)
            with httpx.Client(transport=httpx.MockTransport(handler)) as client,patch('src.scholarly_sources.time.sleep'):
                result=resolve_reference(ref,True,client,{})
            self.assertEqual(len(calls),count);self.assertIn('rate limit',result['verification_status'])

if __name__=='__main__':unittest.main()
