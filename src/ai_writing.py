"""Optional consent-gated GPTZero adapter; probabilities are never text fractions."""
import math
import os
import httpx
from src.research_integrity import writing_observations
from src.scholarly_sources import external_status

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
    settings={key:os.getenv(key,'').strip() for key in ('GPTZERO_API_KEY','GPTZERO_MODEL_VERSION')}
    try:
        import streamlit as st
        for key in settings:
            if not settings[key]:
                settings[key]=str(st.secrets.get(key,'')).strip()
    except Exception:
        pass
    return settings


def assess_writing(paper,consent=False,settings=None,client=None):
    base=writing_observations(paper)
    settings=detector_settings() if settings is None else settings
    if not settings.get('GPTZERO_API_KEY') or not settings.get('GPTZERO_MODEL_VERSION'):
        base['explanation']+=' No supported detector key and explicit model/version are configured.'
        return base
    if not consent:
        base['explanation']+=' External detector not contacted: explicit consent was not granted.'
        return base
    if not paper.body.strip():
        base['explanation']+=' No readable body text is available.'
        return base
    if len(paper.body)>100_000:
        base['explanation']+=' Detector not contacted: body exceeds the local 100,000-character safety limit; no silent truncation.'
        return base
    if client is None:
        with httpx.Client(timeout=30,follow_redirects=False) as owned:
            return assess_writing(paper,True,settings,owned)
    base['external_submission_attempted']=True
    base['characters_attempted']=len(paper.body)
    try:
        response=client.post('https://api.gptzero.me/v2/predict/text',
                             headers={'x-api-key':settings['GPTZERO_API_KEY'],'Accept':'application/json'},
                             json={'document':paper.body,'version':settings['GPTZERO_MODEL_VERSION']})
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
        # Preserve provider sentence scores as raw fields, with no invented meaning.
        sentences=[]
        for item in document.get('sentences',[]):
            if not isinstance(item,dict):
                continue
            text=item.get('sentence',item.get('text',''))
            if not isinstance(text,str) or not text or text not in paper.body:
                continue
            raw_scores={k:v for k,v in item.items() if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)}
            sentences.append(dict(text=text,provider_raw_scores=raw_scores,
                                  meaning='Provider sentence fields, not percentage of text written by AI'))
        return dict(status='External detector assessment — not authorship proof',provider='GPTZero',
                    requested_version=settings['GPTZERO_MODEL_VERSION'],
                    reported_version=data.get('version',document.get('version','Not returned by provider')),
                    classification=classification,class_probabilities=probabilities,
                    confidence_category=document.get('confidence_category','Not returned'),sentences=sentences,
                    explanation=DETECTOR_LIMITATIONS,documentation=PROVIDER_DOC,characters_sent=len(paper.body))
    except Exception as error:
        base['explanation']+=' GPTZero: '+external_status(error)+'. No detector result was inferred.'
        return base
