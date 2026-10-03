"""Session-private Streamlit integrity workflow, separate from Gemini/corpus caches."""
import hashlib
import json
import streamlit as st
from src.research_integrity import (
    extract_pdf, citation_analysis, compare_sources, integrity_report,
    LIMITATIONS, METHOD, MAX_SOURCES,
)
from src.scholarly_sources import resolve_reference, retrieve_open_access, MAX_METADATA_REFERENCES
from src.ai_writing import assess_writing, detector_settings


@st.cache_resource(show_spinner=False)
def _offline_semantic_model():
    from sentence_transformers import SentenceTransformer
    # The model resource is shared, never uploaded documents/embeddings/results.
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',local_files_only=True)


def render_integrity_analysis():
    st.markdown('### Research Paper Integrity Analysis')
    st.caption('Separate source evidence, text similarity, and AI-writing assessment. No result establishes academic misconduct.')
    nonce=st.session_state.setdefault('integrity_upload_nonce',0)
    paper_file=st.file_uploader('Research paper PDF',type=['pdf'],key=f'integrity-paper-{nonce}')
    reference_files=st.file_uploader('Optional reference paper PDFs',type=['pdf'],accept_multiple_files=True,key=f'integrity-references-{nonce}')
    st.caption('Private by default: extraction and uploaded-reference comparisons run locally on the app server. Papers are not added to the corpus, sent to Gemini, or saved in the document disk cache. Results stay in your Streamlit session.')
    if st.button('Clear private integrity uploads and results',key='integrity-clear'):
        st.session_state.pop('integrity_private',None)
        st.session_state.integrity_upload_nonce=nonce+1
        st.rerun()
    if paper_file is None:
        st.session_state.pop('integrity_private',None)
        st.info('Upload a paper here to begin. The existing document corpus and Executive Summary remain separate.')
        return
    data=paper_file.getvalue()
    digest=hashlib.sha256(data).hexdigest()
    scope=hashlib.sha256((digest+''.join(hashlib.sha256(f.getvalue()).hexdigest() for f in reference_files)).encode()).hexdigest()
    state=st.session_state.get('integrity_private')
    if not state or state['scope']!=scope:
        state={'scope':scope,'paper':None,'results':{},'references':{}}
        st.session_state.integrity_private=state
    if state['paper'] is None:
        try:
            with st.spinner('Extracting paper text locally; scanned pages may require OCR…'):
                state['paper']=extract_pdf(data,paper_file.name)
        except Exception as error:
            st.error(f'Paper extraction failed ({type(error).__name__}). Check PDF readability, encryption, size (25 MB) and page count (300).')
            return
    paper=state['paper']
    paper.name=paper_file.name
    citations=citation_analysis(paper)
    st.caption(f'{len(paper.pages)} pages · {len(paper.bibliography)} parsed bibliography entries · {len(citations)} citation markers. Supported citation styles: numbered brackets/ranges and common author-year forms.')
    if not paper.bibliography_found:
        st.warning('No supported bibliography heading found. Bibliography exclusion and citation mapping may be incomplete; review the extracted text.')
    with st.expander('Review extracted text and bibliography'):
        st.text_area('Non-bibliography text',paper.body,height=220,disabled=True,key=f'integrity-preview-{scope}')
        for entry in paper.bibliography:
            st.text(f"{entry['id']} · bibliography pages {entry['pages']} · {entry['raw']}")
        for note in paper.extraction_notes:
            st.caption(note)
    associations={}
    for i,file in enumerate(reference_files):
        options=['No explicit association']+[r['id'] for r in paper.bibliography]
        associations[i]=st.selectbox(f'Bibliography association for {file.name} (optional; confirm this is the same source)',options,
                                     key=f'integrity-association-{scope}-{i}',
                                     help='No source is treated as cited because its subject is similar. Your explicit selection is recorded; no automatic association from topics or reference-paper bibliography DOIs is made.')
    semantic=st.checkbox('Also check semantic similarity with a locally cached MiniLM model',key=f'integrity-semantic-{scope}',
                         help='Offline model only; no model download or text transfer. Cosine similarity is exploratory and never enters the overlap percentage.')
    st.markdown('#### Optional external services — explicit consent')
    metadata_consent=st.checkbox('I consent to sending reference DOIs and up to 1,000 characters per bibliography entry to Crossref for metadata lookup (maximum 25 entries).',key=f'integrity-metadata-consent-{scope}')
    fulltext_consent=st.checkbox('I consent to sending verified reference DOI identifiers to Europe PMC to retrieve available open-access full text (maximum 5 sources). No paper body or passages are sent.',key=f'integrity-fulltext-consent-{scope}')
    settings=detector_settings()
    configured=bool(settings.get('GPTZERO_API_KEY') and settings.get('GPTZERO_MODEL_VERSION'))
    detector_consent=st.checkbox('I consent to sending this paper’s non-bibliography text to GPTZero for the configured external AI-writing assessment.',
                                 key=f'integrity-detector-consent-{scope}',disabled=not configured)
    if not configured:
        st.caption('AI authorship assessment unavailable: no supported detector key and explicit model/version configured. Local writing observations remain available.')
    else:
        st.caption(f"Configured detector: GPTZero · requested model/version: {settings['GPTZERO_MODEL_VERSION']}. This may consume your provider quota; probability is not a percentage written by AI.")
    if fulltext_consent and not metadata_consent:
        st.info('Europe PMC comparison also requires Crossref metadata consent so exact reference DOIs can be verified first.')
    options=dict(associations=associations,semantic=semantic,metadata=metadata_consent,
                 fulltext=fulltext_consent,detector=detector_consent,detector_version=settings.get('GPTZERO_MODEL_VERSION',''))
    result_key=hashlib.sha256(json.dumps(options,sort_keys=True).encode()).hexdigest()
    if st.button('Run Research Paper Integrity Analysis',key='integrity-run',type='primary',disabled=not paper.body.strip()):
        try:
            with st.status('Analyzing evidence…',expanded=True) as status:
                sources=[]
                # No external operations occur until this explicit action AND consent.
                for i,file in enumerate(reference_files):
                    if i>=MAX_SOURCES:
                        sources.append(dict(name=file.name,paper=None,status='Not checked: 10-upload comparison limit'))
                        continue
                    status.update(label=f'Extracting reference {i+1} / {min(len(reference_files),MAX_SOURCES)} locally')
                    fingerprint=hashlib.sha256(file.getvalue()).hexdigest()
                    if fingerprint not in state['references']:
                        try:
                            state['references'][fingerprint]=extract_pdf(file.getvalue(),file.name)
                        except Exception:
                            state['references'][fingerprint]=None
                    other=state['references'][fingerprint]
                    refs=[]
                    association=associations[i]
                    if association!='No explicit association':
                        refs=[association]
                    sources.append(dict(name=file.name,paper=other,reference_ids=refs,
                                        origin='User-uploaded reference; '+('user-confirmed bibliography association' if association!='No explicit association' else 'no cited-source association established'),
                                        status='Extracted locally' if other else 'Unavailable: reference PDF could not be extracted',link=''))
                references=[dict(r) for r in paper.bibliography]
                if metadata_consent:
                    for i,entry in enumerate(references):
                        if i<MAX_METADATA_REFERENCES:
                            status.update(label=f'Crossref metadata {i+1} / {min(len(references),MAX_METADATA_REFERENCES)}')
                            references[i]=resolve_reference(entry,consent=True)
                        else:
                            references[i]['verification_status']='Not checked: 25-reference metadata limit'
                if fulltext_consent and metadata_consent:
                    verified=[r for r in references if r['verification_status'].startswith('DOI record verified')]
                    retrieval_budget=min(5,max(0,MAX_SOURCES-len(reference_files)))
                    for i,entry in enumerate(verified):
                        if i<retrieval_budget:
                            status.update(label=f'Europe PMC open-access lookup {i+1} / {min(len(verified),retrieval_budget)}')
                            sources.append(retrieve_open_access(entry,consent=True))
                        else:
                            sources.append(dict(name=entry['title'] or entry['id'],link=entry['link'],paper=None,
                                                origin='Author bibliography',status='Not checked: external retrieval or total 10-source comparison limit'))
                    for entry in references:
                        if not entry['verification_status'].startswith('DOI record verified'):
                            sources.append(dict(name=entry['title'] or entry['id'],link=entry['link'],paper=None,
                                                origin='Author bibliography',status='Not checked: reference DOI not verified'))
                else:
                    sources.append(dict(name='External full-text sources',paper=None,origin='Europe PMC',
                                        status='Not checked: external full-text retrieval consent not enabled'))
                model=None
                semantic_note=''
                if semantic:
                    try:
                        status.update(label='Loading semantic model locally (no downloads)')
                        model=_offline_semantic_model()
                    except Exception:
                        semantic_note='Unavailable: local MiniLM model not cached; no model download attempted'
                status.update(label='Comparing actual source passages locally')
                analysis=compare_sources(paper,sources,citations,model)
                analysis['consents']={'Crossref bibliography metadata':metadata_consent,
                                      'Europe PMC reference DOI lookup':fulltext_consent and metadata_consent,
                                      'GPTZero non-bibliography body submission':detector_consent}
                if semantic_note:
                    analysis['semantic_status']=semantic_note
                status.update(label='Preparing separate AI-writing assessment')
                ai=assess_writing(paper,consent=detector_consent,settings=settings)
                # Use original parsed reference identity for citation mapping, resolved
                # provider metadata only for display. Preserve raw bibliography pages.
                state['results'][result_key]=dict(references=references,citations=citations,analysis=analysis,ai=ai)
                status.update(label='Integrity analysis complete',state='complete',expanded=False)
        except Exception as error:
            st.error(f'Integrity analysis could not complete ({type(error).__name__}). No result was inferred; try fewer readable reference PDFs.')
    result=state['results'].get(result_key)
    if not result:
        st.info('Run analysis for the current uploads, associations and consent settings. Changing consent hides results from the previous configuration.')
        return
    sources_tab,similarity_tab,ai_tab=st.tabs(['Source and citation analysis','Text similarity','AI-writing assessment'])
    with sources_tab:
        rows=[]
        for ref in result['references']:
            rows.append(dict(Reference=ref['id'],Title=ref.get('title') or 'Unparsed',Authors=ref.get('authors') or 'Unparsed',Year=ref.get('year') or 'Unparsed',
                             DOI=ref.get('doi') or '',Link=ref.get('link') or '',
                             Source_type='Author-cited reference' if ref['cited'] else 'Ambiguous in-text citation candidate; identity not established' if ref.get('ambiguous_citation') else 'Author bibliography; no matched in-text citation',
                             Verification=ref['verification_status'],Bibliography_pages=str(ref['pages'])))
        if rows:
            st.dataframe(rows,use_container_width=True,hide_index=True,column_config={'Link':st.column_config.LinkColumn('Source link')})
        else:
            st.info('No bibliography entries were parsed.')
        for ref in result['references']:
            if ref.get('metadata_candidates'):
                with st.expander(f"{ref['id']} — unverified Crossref candidates (not claimed as sources used)"):
                    st.dataframe(ref['metadata_candidates'],hide_index=True,use_container_width=True,column_config={'link':st.column_config.LinkColumn('Candidate link')})
        st.dataframe([dict(Citation=c['text'],Pages=str(c['pages']),References=', '.join(c['reference_ids']),Status=c['status']) for c in result['citations']],hide_index=True,use_container_width=True)
        missing=[c for c in result['citations'] if c['status']=='Missing bibliography reference']
        if missing:
            st.warning(f'{len(missing)} citation markers lack a mapped bibliography entry. Verify extraction and citation style before treating them as missing references.')
        st.caption('A verified DOI record establishes that the record exists, not that the bibliography description is accurate or that the paper actually used it. Similar topics never establish source use.')
    with similarity_tab:
        analysis=result['analysis']
        a,b=st.columns(2)
        a.metric('Similarity within checked sources',f"{analysis['overlap_percentage']:.2f}%")
        b.metric('Potentially unattributed overlap within checked sources',f"{analysis['unattributed_overlap_percentage']:.2f}%")
        st.caption(f"Quoted and source-cited overlap: {analysis['quoted_cited_overlap_percentage']:.2f}% (subset of total). Quotation alone does not establish attribution.")
        st.caption(f"Denominator: {analysis['analyzed_body_words']} examined non-bibliography word tokens; {analysis['overlapping_words']} unique overlapping tokens. Coverage: {analysis['analyzed_body_words']} / {analysis['total_body_words']} extracted body tokens examined. Bibliography excluded where detected.")
        st.warning('This is not a plagiarism score. A zero result does not imply originality; neither overlap nor semantic similarity establishes academic misconduct.')
        st.dataframe(analysis['coverage'],hide_index=True,use_container_width=True,column_config={'link':st.column_config.LinkColumn('Source link')})
        st.caption('Semantic comparison: '+analysis['semantic_status'])
        st.caption(analysis['candidate_limits'])
        match_count=len(analysis['matches'])
        page_count=max(1,(match_count+19)//20)
        evidence_page=st.number_input('Evidence page (20 matches per page)',min_value=1,max_value=page_count,value=1,step=1,
                                      key=f'integrity-evidence-page-{scope}-{result_key}-{match_count}')
        first=(evidence_page-1)*20
        if match_count:
            st.caption(f'Displaying matches {first+1}–{min(first+20,match_count)} of {match_count}. The downloaded report includes all detected matches; percentages use all matches.')
        for i,match in enumerate(analysis['matches'][first:first+20],start=first):
            with st.expander(f"{match['match_type']} · {match['source_name']} · submitted pages {match['submitted_pages']}"):
                st.text(match['attribution'])
                st.caption(match['source_origin'])
                st.caption('Author-cited source association observed' if match['cited'] else 'Text-matched source; author use not established')
                st.caption(f"Source pages: {match['source_pages'] or 'Not available for full-text XML'} · score: {match['similarity']} ({'cosine, not overlap percentage' if match['match_type']=='semantic similarity' else 'normalized token sequence ratio'})")
                if match['source_link']:
                    st.link_button('Open matching source',match['source_link'])
                else:
                    st.caption('Source is a user-uploaded PDF; no public source link available.')
                left,right=st.columns(2)
                left.text_area('Submitted passage',match['submitted_passage'],height=150,disabled=True,key=f'integrity-left-{scope}-{result_key}-{i}')
                right.text_area('Matching source passage',match['source_passage'],height=150,disabled=True,key=f'integrity-right-{scope}-{result_key}-{i}')
        if not analysis['matches']:
            st.info('No matches passed the stated thresholds in the sources and text actually checked.')
        with st.expander('Methodology and limitations'):
            st.write(METHOD)
            st.write(LIMITATIONS)
    with ai_tab:
        ai=result['ai']
        st.markdown('#### '+ai['status'])
        st.write(ai['explanation'])
        if ai.get('provider'):
            st.caption(f"GPTZero · requested version {ai['requested_version']} · reported version {ai['reported_version']} · classification {ai['classification']} · confidence {ai['confidence_category']}")
            st.dataframe([dict(Class=key,Probability=value,Meaning='Provider classification confidence; not fraction of paper written by AI') for key,value in ai['class_probabilities'].items()],hide_index=True)
            st.link_button('Provider score interpretation',ai['documentation'])
            if ai['sentences']:
                st.caption('Passage-level fields returned by the provider; raw values, not AI-written text percentages.')
                st.dataframe(ai['sentences'],hide_index=True,use_container_width=True)
            else:
                st.caption('Passage-level results were not available in the validated provider response.')
        else:
            st.markdown('##### Local writing observations — not authorship evidence')
            st.json(ai['observations'])
        st.info('AI-writing assessment is independent of source similarity. Neither result proves misconduct.')
    # Construct from results already in session; downloading never calls a provider.
    import copy
    report_paper=copy.copy(paper)
    report_paper.bibliography=result['references']
    st.download_button('Download integrity analysis report',integrity_report(report_paper,result['citations'],result['analysis'],result['ai']),
                       file_name='research-paper-integrity-analysis.md',mime='text/markdown',key=f'integrity-report-{scope}-{result_key}')
