"""Real public-PDF regression; never stores a paper in Git or prints its text.

Supply a local PDF path. Tested with https://arxiv.org/pdf/quant-ph/9708022;
this public edition is not confirmed identical to the user's faulty report input.
"""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import extract_pdf,citation_analysis
p=extract_pdf(Path(sys.argv[1]).read_bytes(),'public Steane regression')
assert p.bibliography_found
assert len(p.bibliography)>=100
assert all(len(r['authors'])<=220 for r in p.bibliography)
assert all(not any(term in r['authors'].casefold() for term in ('introduction','classical information theory')) for r in p.bibliography)
assert all(r['authors']=='' and r['title']=='' and r['year']=='' for r in p.bibliography if not r['lookup_eligible'])
print(json.dumps(dict(pages=len(p.pages),entries=len(p.bibliography),parsed=sum(r['lookup_eligible'] for r in p.bibliography),
    unparsed=sum(not r['lookup_eligible'] for r in p.bibliography),reference_pages=sorted(set(n for r in p.bibliography for n in r['pages'])),
    matched_citation_markers=sum(bool(c['reference_ids']) for c in citation_analysis(p)),
    maximum_author_field_characters=max(len(r['authors']) for r in p.bibliography))))
