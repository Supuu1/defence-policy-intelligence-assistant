"""Public synthetic smoke test; reports metrics only, never paper text."""
import sys,time,json,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.local_ai_writing import assess_local_writing,load_detector,split_passages
from src.research_integrity import paper_from_pages
start=time.perf_counter()
paper=paper_from_pages('synthetic smoke test',[dict(page_number=1,text=('This controlled software test checks whether a local classifier returns a finite score. '
    'The example contains no private document or personal information. '*12))])
result=assess_local_writing(paper)
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# macOS reports bytes; Linux reports KiB.
print(json.dumps(dict(model=result['model'],revision=result['version'],error=result.get('error'),
    analyzed_words=result['analyzed_words'],passage_scores=[s['score'] for s in result['sentences']],
    elapsed_seconds=round(time.perf_counter()-start,2),peak_rss_mib=round(rss/(1024**2 if sys.platform=='darwin' else 1024),1))))
if not result['analyzed_words']:sys.exit(1)
first=load_detector();assert load_detector() is first
long_paper=paper_from_pages('synthetic long test',[dict(page_number=i+1,text=('A controlled public test passage checks token limits and CPU inference. '*70)) for i in range(2)])
long_result=assess_local_writing(long_paper,resource=first)
assert long_result['coverage']['percent']==100
assert len(long_result['sentences'])>1
assert all(s['model_tokens']<=512 for s in long_result['sentences'])
print(json.dumps(dict(long_input_analyzed_words=long_result['analyzed_words'],passages=len(long_result['sentences']),
    maximum_model_tokens=max(s['model_tokens'] for s in long_result['sentences']),coverage_percent=long_result['coverage']['percent'],cached_resource_reused=True)))
