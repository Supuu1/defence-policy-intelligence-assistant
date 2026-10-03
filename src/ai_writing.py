"""Optional consent-gated GPTZero adapter; probabilities are never text fractions."""
import math
import os
import html
import logging
import re
from bisect import bisect_left
import httpx
from src.research_integrity import writing_observations, tokens, pages_at

PROVIDER_DOC='https://support.gptzero.me/articles/8947054519-how-do-i-use-and-interpret-the-results-from-your-api'
DETECTOR_LIMITATIONS=(
    'Provider class probabilities express confidence in HUMAN_ONLY, MIXED, or AI_ONLY '
    'classification for similar documents, not the proportion of this paper written by AI. '
    'False positives/negatives, editing, translation, language and genre limit inference. '
    'Text alone cannot establish authorship or academic misconduct.'
)


def detector_settings():
    from dotenv import load_dotenv
    load_dotenv()
    settings={key:os.getenv(key,'').strip() for key in ('GPTZERO_API_KEY','GPTZERO_MODEL_VERSION','GPTZERO_MAX_CHARACTERS')}
    try:
        import streamlit as st
        for key in settings:
            if not settings[key]:
                settings[key]=str(st.secrets.get(key,'')).strip()
    except Exception:
        pass
    return settings


def input_limit(settings):
    """Account-confirmed request limit; no guessed provider maximum."""
    try:
        value=int(settings.get('GPTZERO_MAX_CHARACTERS',''))
        return value if 0 < value <= 1_000_000 else None
    except (TypeError,ValueError):
        return None


def passage_results(paper, submitted, items):
    """Exact unique alignment only; ambiguous repeated passages are not inferred."""
    words=tokens(submitted)
    starts=[a for _,a,_ in words]
    flagged=set()
    covered=set()
    sentences=[]
    seen=set()
    unaligned=0
    flags_present=False
    for item in items:
        if not isinstance(item,dict):
            unaligned+=1
            continue
        text=item.get('sentence',item.get('text',''))
        flag=item.get('highlight_sentence_for_ai')
        flags_present |= isinstance(flag,bool)
        if not isinstance(text,str) or not text or submitted.count(text)!=1:
            unaligned+=1
            continue
        start=submitted.index(text); end=start+len(text)
        identity=(start,end,flag if isinstance(flag,bool) else None)
        if identity in seen:
            continue
        seen.add(identity)
        left=bisect_left(starts,start); right=bisect_left(starts,end)
        indices={i for i in range(left,right) if words[i][2]<=end}
        if isinstance(flag,bool):
            covered.update(indices)
            if flag:
                flagged.update(indices)
        raw={k:v for k,v in item.items() if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)}
        sentences.append(dict(text=text,start=start,end=end,pages=pages_at(paper,start,end),
                              flagged=flag if isinstance(flag,bool) else None,provider_raw_scores=raw))
    complete=bool(words) and len(covered)==len(words) and unaligned==0
    return dict(sentences=sentences,flagged_words=len(flagged),analyzed_words=len(words),
                passage_covered_words=len(covered),unaligned_passages=unaligned,
                flagged_percentage=100*len(flagged)/len(words) if flags_present and complete else None,
                passage_note=('Provider sentence flags cover all analyzed words.' if complete else
                    'No text percentage: sentence flags are absent, incomplete, or cannot be uniquely aligned. Valid flags are still highlighted.'),
                threshold='Provider highlight_sentence_for_ai is exactly true; no application probability threshold.',
                percentage_method='Union of flagged analyzed word-token positions / all submitted non-bibliography word tokens × 100; overlapping positions count once. Percentage is shown only for complete validated sentence-flag coverage.')


def highlighted_passage(sentence):
    # Escape all paper text before adding the application's fixed markup.
    text=html.escape(sentence['text'])
    content='<mark>'+text+'</mark>' if sentence.get('flagged') is True else text
    # A block HTML wrapper also prevents Markdown image/link syntax being interpreted.
    return '<div style="white-space:pre-wrap">'+content+'</div>'


def detector_error(error):
    """Return actionable fixed messages, never private provider payloads."""
    if isinstance(error,httpx.HTTPStatusError):
        status=error.response.status_code
        if status in (401,403):
            return 'Authentication/access rejected. Check GPTZERO_API_KEY and active API subscription permissions.'
        if status in (402,429):
            return 'Quota/payment or rate limit reached. Check API usage, billing and request allowance before running again; no automatic retries.'
        if status in (400,404,413,422):
            return 'Unsupported request/configuration. Confirm GPTZERO_MODEL_VERSION and GPTZERO_MAX_CHARACTERS with your API account; check readable input.'
        if status>=500:
            return 'Detector service failed. Try again later; no assessment is inferred.'
        return f'Detector rejected the request (HTTP {status}). Check API account configuration.'
    if isinstance(error,httpx.TimeoutException):
        return 'Detector request timed out. Try again later; the provider may already have charged quota.'
    if isinstance(error,httpx.RequestError):
        return 'Cannot reach the detector. Check outbound HTTPS/network connectivity.'
    return 'Malformed or unsupported detector response. No score was inferred; check the configured version against official API documentation.'


def assess_writing(paper,consent=False,settings=None,client=None):
    base=writing_observations(paper)
    settings=detector_settings() if settings is None else settings
    if not settings.get('GPTZERO_API_KEY') or not settings.get('GPTZERO_MODEL_VERSION'):
        base['explanation']+=' Configure GPTZERO_API_KEY and an account-supported GPTZERO_MODEL_VERSION in Streamlit Secrets or environment variables.'
        return base
    if not consent:
        base['explanation']+=' External detector not contacted: explicit consent was not granted.'
        return base
    if not paper.body.strip() or not tokens(paper.body):
        base['explanation']+=' No readable body text is available.'
        return base
    limit=input_limit(settings)
    if limit is None:
        base['explanation']+=' Configure GPTZERO_MAX_CHARACTERS to the positive per-request character limit confirmed in your GPTZero API account (local ceiling: 1,000,000).'
        return base
    # A single bounded excerpt, no undocumented chunk probability aggregation.
    submitted=paper.body[:limit]
    if len(submitted)<len(paper.body):
        boundaries=list(re.finditer(r'\s+',submitted))
        if boundaries:
            submitted=submitted[:boundaries[-1].start()]
        else:
            base['explanation']+=' No complete word fits the configured excerpt limit. Increase the account-confirmed limit or check extracted text.'
            return base
    if client is None:
        with httpx.Client(timeout=30,follow_redirects=False) as owned:
            return assess_writing(paper,True,settings,owned)
    base['external_submission_attempted']=True
    base['characters_attempted']=len(submitted)
    try:
        response=client.post('https://api.gptzero.me/v2/predict/text',
                             headers={'x-api-key':settings['GPTZERO_API_KEY'],'Accept':'application/json'},
                             json={'document':submitted,'version':settings['GPTZERO_MODEL_VERSION']})
        response.raise_for_status()
        data=response.json()
        document=data['documents'][0]
        probabilities=document['class_probabilities']
        if not isinstance(probabilities,dict) or not probabilities:
            raise ValueError('Missing documented class probabilities')
        for key,value in probabilities.items():
            if key not in {'human','mixed','ai'} or isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Invalid probability')
        classification=document['document_classification']
        if classification not in {'HUMAN_ONLY','MIXED','AI_ONLY'}:
            raise ValueError('Unknown detector classification')
        items=document.get('sentences',[])
        if not isinstance(items,list):
            raise ValueError('Invalid sentence results')
        passages=passage_results(paper,submitted,items)
        verdict={'HUMAN_ONLY':'Likely human-written','MIXED':'Mixed','AI_ONLY':'Likely AI-generated'}[classification]
        total_words=len(tokens(paper.body))
        return dict(status='External detector assessment — not authorship proof',provider='GPTZero',
                    requested_version=settings['GPTZERO_MODEL_VERSION'],
                    reported_version=data.get('version',document.get('version','Not returned by provider')),
                    classification=classification,verdict=verdict,class_probabilities=probabilities,
                    ai_generation_probability=probabilities.get('ai'),
                    coverage=dict(analyzed_words=len(tokens(submitted)),total_body_words=total_words,
                                  percent=100*len(tokens(submitted))/total_words if total_words else 0,
                                  characters=len(submitted),total_body_characters=len(paper.body),
                                  pages=pages_at(paper,0,len(submitted)),partial=len(submitted)<len(paper.body)),
                    confidence_category=document.get('confidence_category','Not returned'),**passages,
                    explanation=DETECTOR_LIMITATIONS,documentation=PROVIDER_DOC,characters_sent=len(submitted))
    except Exception as error:
        status=error.response.status_code if isinstance(error,httpx.HTTPStatusError) else None
        logging.getLogger(__name__).warning('GPTZero request failed: exception_class=%s http_status=%s',type(error).__name__,status)
        base['explanation']+=' GPTZero: '+detector_error(error)+'. No detector result was inferred.'
        return base
