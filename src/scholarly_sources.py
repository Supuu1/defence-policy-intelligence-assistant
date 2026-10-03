"""Consent-gated Crossref metadata and Europe PMC open-access full text.

No arbitrary URL fetching, publisher scraping, paywall bypass or paper-text search.
Provider errors are classified without logging request URLs, content or secrets.
"""
from datetime import datetime, timezone
from urllib.parse import quote
import re
import xml.etree.ElementTree as ET

import httpx
from src.research_integrity import paper_from_pages, doi_link

CROSSREF = 'https://api.crossref.org'
EUROPE_PMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest'
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_METADATA_REFERENCES = 25


class ExternalSourceError(RuntimeError):
    pass


def external_status(error):
    if isinstance(error, httpx.TimeoutException):
        return 'Unavailable: external service timeout'
    if isinstance(error, httpx.HTTPStatusError):
        code=error.response.status_code
        return {401:'Unavailable: service authentication required',403:'Unavailable: service access denied',
                404:'Not found in service',429:'Unavailable: service rate limit'}.get(code,f'Unavailable: service HTTP {code}')
    if isinstance(error, httpx.NetworkError):
        return 'Unavailable: external service network failure'
    return 'Unavailable: malformed or unsupported external response'


def _request(client,path,params=None):
    # Fixed provider URLs only; HTTP redirects are never followed.
    with client.stream('GET',path,params=params) as response:
        response.raise_for_status()
        data=bytearray()
        for piece in response.iter_bytes():
            data.extend(piece)
            if len(data)>MAX_RESPONSE_BYTES:
                raise ExternalSourceError('External response exceeds the 8 MB limit')
        return bytes(data)


def _json(client,path,params=None):
    import json
    return json.loads(_request(client,path,params))


def _metadata(item):
    title=(item.get('title') or [''])[0]
    authors=', '.join(' '.join(filter(None,(a.get('given',''),a.get('family','')))) for a in item.get('author',[]))
    dates=item.get('published',item.get('issued',{})).get('date-parts',[])
    year=str(dates[0][0]) if dates and dates[0] else ''
    doi=item.get('DOI','').lower()
    return dict(title=title,authors=authors,year=year,doi=doi,link=doi_link(doi))


def resolve_reference(reference,consent=False,client=None):
    if not consent:
        raise PermissionError('Explicit consent is required before sending bibliography details to Crossref.')
    if client is None:
        with httpx.Client(timeout=15,follow_redirects=False,headers={'User-Agent':'ResearchIntegrityAssistant/1.0'}) as owned:
            return resolve_reference(reference,True,owned)
    result=dict(reference)
    result['metadata_checked_at']=datetime.now(timezone.utc).isoformat()
    try:
        if reference.get('doi'):
            data=_json(client,f"{CROSSREF}/works/{quote(reference['doi'],safe='')}")
            metadata=_metadata(data['message'])
            if metadata['doi']!=reference['doi'].lower():
                raise ExternalSourceError('DOI response mismatch')
            result.update(metadata)
            result['verification_status']='DOI record verified in Crossref; bibliography details still require review'
        else:
            data=_json(client,f'{CROSSREF}/works',{'query.bibliographic':reference['raw'][:1000],'rows':3})
            candidates=[_metadata(item) for item in data['message'].get('items',[])]
            # Search relevance is not evidence of a reference's identity or use.
            result['metadata_candidates']=candidates
            result['verification_status']='Candidate metadata only — not verified; review title/authors/year'
            if not candidates:
                result['verification_status']='No Crossref candidate found'
        return result
    except Exception as error:
        result['verification_status']=external_status(error)
        return result


def resolve_bibliography(references,consent=False,client=None):
    if not consent:
        raise PermissionError('Explicit metadata lookup consent is required.')
    result=[]
    for i,reference in enumerate(references):
        if i>=MAX_METADATA_REFERENCES:
            result.append(dict(reference,verification_status='Not checked: 25-reference lookup limit'))
        else:
            result.append(resolve_reference(reference,True,client))
    return result


def retrieve_open_access(reference,consent=False,client=None):
    if not consent:
        raise PermissionError('Explicit consent is required before sending reference DOI identifiers to Europe PMC.')
    source=dict(name=reference.get('title') or reference['id'],link=reference.get('link',''),
                origin='Europe PMC open-access text; associated author bibliography DOI',
                reference_ids=[reference['id']],status='Unavailable: no verified DOI',paper=None)
    # Only DOI-verified records are used. Search candidates never become cited sources.
    if not reference.get('doi') or not reference.get('verification_status','').startswith('DOI record verified'):
        return source
    if client is None:
        with httpx.Client(timeout=20,follow_redirects=False) as owned:
            return retrieve_open_access(reference,True,owned)
    try:
        data=_json(client,f'{EUROPE_PMC}/search',{'query':f'DOI:"{reference["doi"]}"','format':'json','resultType':'core','pageSize':5})
        results=data.get('resultList',{}).get('result',[])
        item=next((r for r in results if r.get('doi','').lower()==reference['doi'].lower() and r.get('isOpenAccess')=='Y' and re.fullmatch(r'PMC\d+',r.get('pmcid',''))),None)
        if not item:
            source['status']='Unavailable: no Europe PMC open-access full text for this DOI'
            return source
        pmcid=item['pmcid']
        raw=_request(client,f'{EUROPE_PMC}/{pmcid}/fullTextXML')
        if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
            # JATS commonly has a harmless external DTD. Strip DOCTYPE declarations
            # with no internal subset; never load or expand external/internal entities.
            if b'<!ENTITY' in raw.upper() or re.search(rb'<!DOCTYPE[^>]*\[',raw,re.I):
                raise ExternalSourceError('Unsupported XML entity declaration')
            raw=re.sub(rb'<!DOCTYPE[^>]*>',b'',raw,flags=re.I)
        root=ET.fromstring(raw)
        body=root.find('.//body')
        if body is None:
            raise ExternalSourceError('No article body')
        paragraphs=[' '.join(p.itertext()).strip() for p in body.findall('.//p')]
        text='\n\n'.join(p for p in paragraphs if p)
        if not text:
            raise ExternalSourceError('Empty open-access article body')
        source.update(paper=paper_from_pages(source['name'],[dict(page_number=None,text=text,extraction_method='Open-access XML')]),
                      link=f'https://europepmc.org/articles/{pmcid}',status='Open-access text retrieved',
                      license=' '.join(root.findtext('.//license-p','').split()) or item.get('license','Not specified in returned record'),
                      retrieved_at=datetime.now(timezone.utc).isoformat())
        return source
    except Exception as error:
        source['status']=external_status(error)
        return source
