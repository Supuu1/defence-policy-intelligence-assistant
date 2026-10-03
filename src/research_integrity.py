"""Deterministic, local research-paper evidence analysis. No generative calls."""
from dataclasses import dataclass
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import re
import unicodedata
from urllib.parse import quote, urlsplit

MAX_SUBMITTED_WORDS = 20_000
MAX_SOURCE_WORDS = 50_000
MAX_SOURCES = 10
MAX_PASSAGES = 1000
MIN_OVERLAP_WORDS = 8
NEAR_EXACT_THRESHOLD = .88
SEMANTIC_THRESHOLD = .82
WORD = re.compile(r"\b\w+(?:['’\-]\w+)*\b", re.UNICODE)
DOI = re.compile(r"\b10\.\d{4,9}/[^\s<>\"]+", re.I)
YEAR = re.compile(r"\b((?:19|20)\d{2}[a-z]?)\b")
METHOD = (
    "Similarity within checked sources: union of submitted word-token positions "
    "identical in exact or near-exact matches / examined non-bibliography word tokens × 100. "
    "Overlapping positions count once across all sources. Quoted/cited matches are included "
    "in total overlap and shown separately. Near-exact substitutions do not count as "
    "overlapping tokens. Normalization is Unicode NFKC, case folding, and word tokenization; "
    "case, punctuation and whitespace differences are ignored. Exact runs require at least "
    "8 words; near-exact passage pairs require token SequenceMatcher ratio ≥ 0.88 and at "
    "least 8 identical tokens. Semantic cosine similarity is separate and never enters "
    "the overlap percentage. No LLM calculates matches, citations or percentages."
)
LIMITATIONS = (
    "This is similarity within checked sources, not a plagiarism score or evidence of "
    "academic misconduct. Zero overlap does not imply originality. Unchecked, paywalled "
    "or unavailable sources, images, formulas, extraction/OCR errors, short overlaps, "
    "paraphrases and unsupported citation styles can be missed. Citation/quotation "
    "association is heuristic and requires review; it does not establish correct quotation "
    "practice. Semantic/topic similarity does not show that the author used a source. "
    "Text alone cannot reliably establish AI authorship or an exact fraction written by AI."
)


def safe_link(value):
    if not isinstance(value, str):
        return ""
    if any(ord(char)<32 for char in value):
        return ''
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ''
    return value if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password else ""


def extract_dois(text):
    return list(dict.fromkeys(m.group(0).rstrip('.,;:)\u005d}').lower() for m in DOI.finditer(text)))


def doi_link(doi):
    return 'https://doi.org/' + quote(doi, safe='/') if doi else ''


@dataclass
class Paper:
    name: str
    digest: str
    pages: list[dict]
    body: str
    page_spans: list[tuple]
    bibliography: list[dict]
    bibliography_found: bool
    extraction_notes: list[str]


def page_at(paper, offset):
    for start, end, number in paper.page_spans:
        if start <= offset < end:
            return number
    return None


def pages_at(paper, start, end):
    return list(dict.fromkeys(number for a, b, number in paper.page_spans if start < b and end > a and number is not None))


def paper_from_pages(name, pages, digest=None):
    """Find bibliography headings conservatively; retain body/page offset mapping."""
    body_parts, spans, reference_lines, notes = [], [], [], []
    in_refs, found, cursor = False, False, 0
    for page in pages:
        text = page.get('text', '') or ''
        number = page.get('page_number')
        if not text.strip():
            notes.append(f"Page {number}: no readable text extracted")
        if page.get('extraction_method') == 'OCR':
            notes.append(f"Page {number}: OCR text; verify citations and punctuation")
        for line in text.splitlines(keepends=True):
            if re.fullmatch(r'\s*(?:\d+[. ]+)?(?:references(?: and notes)?|bibliography|works cited|literature cited|reference list)\s*\n?', line, re.I):
                in_refs, found = True, True
                continue
            if in_refs and re.match(r'^\s*(?:appendix\b|appendices\b|supplementary (?:material|information)\b)', line, re.I):
                in_refs = False
            if in_refs:
                reference_lines.append((line.strip(), number))
            else:
                body_parts.append(line)
                spans.append((cursor, cursor + len(line), number))
                cursor += len(line)
        body_parts.append('\n')
        spans.append((cursor, cursor + 1, number))
        cursor += 1
    entries, current, ref_pages, label = [], [], [], None
    def flush():
        if not current:
            return
        raw = ' '.join(current).strip()
        year_match = YEAR.search(raw)
        year = year_match.group(1) if year_match else ''
        prefix = raw[:year_match.start()] if year_match else raw
        surname = re.search(r"\b([A-ZÀ-ÖØ-Þ][\w’'\-]+)\s*(?:,|\bet al\b|\band\b|&)", prefix)
        if not surname:
            surname = re.match(r"([A-ZÀ-ÖØ-Þ][\w’'\-]+)\b", prefix)
        # Title inference is marked parsed/unverified, never scholarly verification.
        tail = raw[year_match.end():].lstrip(')., :') if year_match else ''
        title = re.split(r'\.\s|https?://|doi:', tail, maxsplit=1, flags=re.I)[0].strip() if tail else ''
        dois = extract_dois(raw)
        entries.append(dict(id=f'R{len(entries)+1}', label=label, raw=raw,
                            pages=list(dict.fromkeys(ref_pages)), year=year,
                            surname=surname.group(1).casefold() if surname else '',
                            title=title, authors=prefix.strip(' (,.'), doi=dois[0] if dois else '',
                            link=doi_link(dois[0]) if dois else '',
                            verification_status='Parsed from bibliography; not externally verified',
                            origin='Author bibliography', cited=False))
    for line, page in reference_lines:
        numbered = re.match(r'^\s*(?:\[(\d+)\]|(\d+)\.)\s*(.*)', line)
        author_start = re.match(r"^[A-ZÀ-ÖØ-Þ][\w’'\-]+,\s*[A-Z]", line)
        if numbered or (author_start and current) or (not line and current):
            flush()
            current, ref_pages, label = [], [], None
        if numbered:
            label = int(numbered.group(1) or numbered.group(2))
            line = numbered.group(3)
        if line:
            current.append(line)
            ref_pages.append(page)
    flush()
    if not found:
        notes.append('No supported bibliography heading found; bibliography exclusion and citation mapping may be incomplete')
    body = ''.join(body_parts)
    digest = digest or hashlib.sha256(('\n'.join(p.get('text', '') or '' for p in pages)).encode()).hexdigest()
    return Paper(name, digest, pages, body, spans, entries, found, notes)


def extract_pdf(data, name):
    """In-memory only: does not use the corpus disk cache or global document state."""
    import fitz
    if len(data) > 25 * 1024 * 1024:
        raise ValueError('PDF exceeds the 25 MB integrity-analysis limit.')
    from src.ocr_service import extract_text_with_ocr
    pages = []
    with fitz.open(stream=data, filetype='pdf') as pdf:
        if pdf.needs_pass:
            raise ValueError('Password-protected PDF cannot be analyzed.')
        if len(pdf) > 300:
            raise ValueError('PDF exceeds the 300-page integrity-analysis limit.')
        for index, page in enumerate(pdf):
            text, method = page.get_text('text'), 'Native text'
            if len(''.join(text.split())) < 40:
                try:
                    recognized = extract_text_with_ocr(page)
                    if recognized:
                        text, method = recognized, 'OCR'
                except Exception:
                    method = 'Native text (OCR unavailable)' if text.strip() else 'Unreadable'
            pages.append(dict(text=text, page_number=index+1, extraction_method=method))
    return paper_from_pages(name, pages, hashlib.sha256(data).hexdigest())


def citation_analysis(paper):
    refs = paper.bibliography
    for ref in refs:
        ref['cited']=False
        ref['ambiguous_citation']=False
    found = []
    def add(raw, start, end, targets, kind):
        ids = [r['id'] for r in targets]
        status = 'Matched bibliography entry' if len(ids) == 1 else ('Ambiguous bibliography match' if ids else 'Missing bibliography reference')
        for target in targets:
            if len(ids)==1:
                target['cited'] = True
            else:
                target['ambiguous_citation']=True
        found.append(dict(text=raw, start=start, end=end, pages=pages_at(paper,start,end),
                          reference_ids=ids, status=status, style=kind))
    for match in re.finditer(r'\[(\d+(?:\s*[,;–\-]\s*\d+)*)\]', paper.body):
        labels = []
        for part in re.split(r'[,;]',match.group(1)):
            limits = re.split(r'[–\-]',part.strip())
            if len(limits)==2:
                first,last=map(int,limits)
                if 0 <= last-first <= 100:
                    labels.extend(range(first,last+1))
            else:
                labels.append(int(limits[0]))
        for label in dict.fromkeys(labels):
            add(f'[{label}]',match.start(),match.end(),[r for r in refs if r['label']==label], 'numeric')
    pattern = r"\b([A-ZÀ-ÖØ-Þ][\w’'\-]+)(?:\s+et\s+al\.?)?(?:\s+(?:and|&)\s+[A-ZÀ-ÖØ-Þ][\w’'\-]+)?\s*,?\s*\(?((?:19|20)\d{2}[a-z]?)\)?"
    for match in re.finditer(pattern,paper.body):
        surname, year = match.group(1).casefold(), match.group(2)
        prefix=paper.body[:match.start()]
        in_parenthesis=prefix.rfind('(')>prefix.rfind(')')
        if not in_parenthesis and ',' not in match.group(0) and '(' not in match.group(0):
            continue
        targets = [r for r in refs if r['surname']==surname and r['year']==year]
        add(match.group(0),match.start(),match.end(),targets,'author-year')
    for match in DOI.finditer(paper.body):
        doi = extract_dois(match.group(0))[0]
        add(match.group(0),match.start(),match.end(),[r for r in refs if r['doi']==doi],'DOI')
    return found


def tokens(text):
    return [(unicodedata.normalize('NFKC',m.group(0)).casefold(),m.start(),m.end()) for m in WORD.finditer(text)]


def merged_ranges(ranges):
    result=[]
    for start,end in sorted(ranges):
        if end<=start:
            continue
        if result and start<=result[-1][1]:
            result[-1]=(result[-1][0],max(end,result[-1][1]))
        else:
            result.append((start,end))
    return result


def overlap_percentage(ranges, denominator):
    return 100 * sum(b-a for a,b in merged_ranges(ranges)) / denominator if denominator else 0.0


def passages(paper, word_tokens):
    """Non-overlapping 60-word windows; exact matching also crosses window boundaries."""
    return [(i,min(i+60,len(word_tokens))) for i in range(0,len(word_tokens),60)
            if min(i+60,len(word_tokens))-i >= MIN_OVERLAP_WORDS][:MAX_PASSAGES]


def _attribution(paper, char_start, char_end, source_ids, citations):
    quotes = [(m.start(),m.end()) for m in re.finditer(r'“[^”]+”|"[^"\n]+"|«[^»]+»|(?m:^\s*>[^\n]+)',paper.body)]
    quoted = any(a<=char_start and char_end<=b for a,b in quotes)
    nearby = [c for c in citations if c['start'] <= char_end+180 and c['end'] >= char_start-80
              and c['status']=='Matched bibliography entry'
              and set(c['reference_ids']) & set(source_ids)
              and '\n\n' not in paper.body[min(c['end'],char_end):max(c['start'],char_start)]]
    cited = bool(nearby)
    if quoted and cited:
        status='Quoted and source citation observed — review quotation accuracy'
    elif cited:
        status='Source citation observed; complete quotation not detected'
    elif quoted:
        status='Quoted; source citation association not established'
    else:
        status='Potentially unattributed overlap — review required'
    return dict(quoted=quoted,cited=cited,attribution=status,citation_text=[c['text'] for c in nearby])


def compare_sources(paper, sources, citations, semantic_model=None):
    target=tokens(paper.body)
    analyzed=target[:MAX_SUBMITTED_WORDS]
    target_words=[x[0] for x in analyzed]
    matches,coverage,seen_hashes=[],[],{paper.digest}
    semantic_status='Not requested'
    left_vectors=None
    semantic_failures=False
    associations=defaultdict(set)
    for item in sources:
        if item.get('paper'):
            associations[item['paper'].digest].update(item.get('reference_ids',[]))
    for source_index, source in enumerate(sources):
        other=source.get('paper')
        row=dict(name=source.get('name',other.name if other else 'Source'),link=safe_link(source.get('link','')),
                 origin=source.get('origin','User-uploaded reference'), status=source.get('status','Unavailable'),
                 words_checked=0,license=source.get('license','User-uploaded reference or no retrieved license'),
                 retrieved_at=source.get('retrieved_at','Not externally retrieved'),
                 semantic_status='Not requested' if semantic_model is None else 'Not checked')
        if not other:
            coverage.append(row); continue
        if source_index>=MAX_SOURCES:
            row['status']='Not checked: 10-source comparison limit'; coverage.append(row); continue
        if other.digest in seen_hashes:
            row['status']='Not checked: duplicate content or submitted paper itself'; coverage.append(row); continue
        seen_hashes.add(other.digest)
        reference=tokens(other.body)[:MAX_SOURCE_WORDS]
        if not reference:
            row['status']='Unavailable: no readable non-bibliography text';coverage.append(row);continue
        row.update(status='Checked',words_checked=len(reference),total_body_words=len(tokens(other.body)),
                   bibliography_found=other.bibliography_found)
        coverage.append(row)
        words=[x[0] for x in reference]
        source_ids=sorted(associations[other.digest])
        def evidence(a,b,c,d,kind,score,ranges):
            start,end=analyzed[a][1],analyzed[b-1][2]
            rs,re_=reference[c][1],reference[d-1][2]
            match=dict(source_name=row['name'],source_id=other.digest,source_link=row['link'],source_origin=row['origin'],
                       source_reference_ids=source_ids,match_type=kind,similarity=round(score,4),
                       submitted_passage=paper.body[start:end],source_passage=other.body[rs:re_],
                       submitted_pages=pages_at(paper,start,end),source_pages=pages_at(other,rs,re_),
                       submitted_word_start=a,submitted_word_end=b,source_word_start=c,source_word_end=d,
                       overlap_ranges=ranges)
            match.update(_attribution(paper,start,end,source_ids,citations))
            return match
        # Indexed 8-word seeds: deterministic exact runs, not an LLM/web plagiarism oracle.
        seeds=defaultdict(list)
        for i in range(max(0,len(words)-MIN_OVERLAP_WORDS+1)):
            key=tuple(words[i:i+MIN_OVERLAP_WORDS])
            if len(seeds[key])<20:
                seeds[key].append(i)
        exact=[]
        last_end=0
        for a in range(max(0,len(target_words)-MIN_OVERLAP_WORDS+1)):
            if a<last_end:
                continue
            candidates=[]
            for c in seeds.get(tuple(target_words[a:a+MIN_OVERLAP_WORDS]),[]):
                b,d=a+MIN_OVERLAP_WORDS,c+MIN_OVERLAP_WORDS
                while b<len(target_words) and d<len(words) and target_words[b]==words[d]:
                    b+=1; d+=1
                candidates.append((a,b,c,d))
            if candidates:
                a,b,c,d=max(candidates,key=lambda x:x[1]-x[0])
                exact.append(evidence(a,b,c,d,'exact overlap',1.0,[(a,b)]))
                last_end=b
        matches.extend(exact)
        target_windows=passages(paper,analyzed)
        source_windows=passages(other,reference)
        # Candidate retrieval by shared words; bounded lexical comparisons.
        index=defaultdict(set)
        for i,(c,d) in enumerate(source_windows):
            for word in set(words[c:d]):
                index[word].add(i)
        near=[]
        for a,b in target_windows:
            counts=defaultdict(int)
            for word in set(target_words[a:b]):
                for i in index.get(word,()):
                    counts[i]+=1
            for i in sorted(counts,key=counts.get,reverse=True)[:10]:
                c,d=source_windows[i]
                ratio=SequenceMatcher(None,target_words[a:b],words[c:d],autojunk=False)
                score=ratio.ratio()
                if NEAR_EXACT_THRESHOLD<=score<1:
                    blocks=ratio.get_matching_blocks()
                    ranges=[(a+block.a,a+block.a+block.size) for block in blocks if block.size]
                    if sum(y-x for x,y in ranges)>=MIN_OVERLAP_WORDS and not any(m['submitted_word_start']<=a and b<=m['submitted_word_end'] for m in exact):
                        near.append(evidence(a,b,c,d,'near-exact overlap',score,ranges))
                        break
        matches.extend(near)
        if semantic_model is not None and target_windows and source_windows:
            import numpy as np
            semantic_status='Checked with local sentence-transformers/all-MiniLM-L6-v2; cosine threshold 0.82; exploratory'
            try:
                if left_vectors is None:
                    left_vectors=semantic_model.encode([paper.body[analyzed[a][1]:analyzed[b-1][2]] for a,b in target_windows],normalize_embeddings=True,show_progress_bar=False)
                left=left_vectors
                right=semantic_model.encode([other.body[reference[c][1]:reference[d-1][2]] for c,d in source_windows],normalize_embeddings=True,show_progress_bar=False)
                row['semantic_status']='Checked locally'
                row['semantic_submitted_windows']=len(target_windows)
                row['semantic_source_windows']=len(source_windows)
                for i,(a,b) in enumerate(target_windows):
                    scores=np.asarray(left[i]) @ np.asarray(right).T
                    best=int(np.argmax(scores)); score=float(scores[best])
                    if score>=SEMANTIC_THRESHOLD and not any(m['submitted_word_start']<=a and b<=m['submitted_word_end'] for m in exact+near):
                        c,d=source_windows[best]
                        matches.append(evidence(a,b,c,d,'semantic similarity',score,[]))
            except Exception:
                semantic_status='Unavailable: local semantic encoding failed; lexical matches retained'
                semantic_failures=True
                row['semantic_status']='Unavailable: local semantic encoding failed'
    if semantic_model is not None and semantic_status=='Not requested':
        semantic_status='No readable checked source/passage pairs for semantic comparison'
    if semantic_failures and any(row['semantic_status']=='Checked locally' for row in coverage):
        semantic_status='Partially available: some source encodings failed; see coverage'
    unique=[]; keys=set()
    for m in matches:
        key=(m['source_id'],m['submitted_word_start'],m['submitted_word_end'],m['source_word_start'],m['source_word_end'],m['match_type'])
        if key not in keys:
            keys.add(key);unique.append(m)
    ranges=[r for m in unique for r in m['overlap_ranges']]
    attributed_positions={i for m in unique if m['cited'] for a,b in m['overlap_ranges'] for i in range(a,b)}
    unattributed=[(i,i+1) for a,b in merged_ranges(ranges) for i in range(a,b) if i not in attributed_positions]
    quoted_cited=[r for m in unique if m['quoted'] and m['cited'] for r in m['overlap_ranges']]
    return dict(matches=unique,coverage=coverage,total_body_words=len(target),analyzed_body_words=len(analyzed),
                overlapping_words=sum(b-a for a,b in merged_ranges(ranges)),
                overlap_percentage=overlap_percentage(ranges,len(analyzed)),
                unattributed_overlap_percentage=overlap_percentage(unattributed,len(analyzed)),
                quoted_cited_overlap_percentage=overlap_percentage(quoted_cited,len(analyzed)),
                semantic_status=semantic_status,methodology=METHOD,limitations=LIMITATIONS,
                candidate_limits='First 20,000 submitted tokens; first 50,000 source tokens; 10 sources; first 1,000 60-word windows; top 10 lexical candidates/window; at most 20 positions per exact seed.')


def writing_observations(paper):
    sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+',paper.body) if len(tokens(s))>=4]
    lengths=[len(tokens(s)) for s in sentences]
    starts=defaultdict(int)
    for sentence in sentences:
        starts[' '.join(x[0] for x in tokens(sentence)[:3])]+=1
    return dict(status='AI authorship assessment unavailable',
                explanation='Text alone cannot reliably establish AI authorship or an exact percentage written by AI. These writing observations are not a detector or proof.',
                observations=dict(sentences_observed=len(sentences),
                                  average_sentence_words=round(sum(lengths)/len(lengths),1) if lengths else None,
                                  sentence_length_range=[min(lengths),max(lengths)] if lengths else [],
                                  repeated_three_word_sentence_openings={k:v for k,v in starts.items() if v>1}))


def integrity_report(paper,citations,analysis,ai):
    def literal(text):
        return '\n'.join('    '+line for line in str(text).splitlines())
    lines=['# Research Paper Integrity Analysis','',f'Paper: {paper.name}',
           '', '## Source and citation analysis','',
           'Author bibliography and in-text citations are distinct from sources checked/discovered through text matching.']
    for ref in paper.bibliography:
        lines.extend(['',f"### {ref['id']} — {ref.get('title') or 'Title unparsed'}",
                      f"Authors: {ref.get('authors') or 'Unparsed'}; year: {ref.get('year') or 'Unparsed'}",
                      f"Origin: {ref['origin']}; explicitly cited: {ref['cited']}; status: {ref['verification_status']}",
                      f"DOI: {ref.get('doi') or 'Not available'}; link: {safe_link(ref.get('link','')) or 'Not available'}",
                      f"Bibliography pages: {ref['pages']}",literal(ref['raw'])])
    for c in citations:
        lines.append(f"- {c['text']} — pages {c['pages']} — {c['status']} — references {c['reference_ids']}")
    lines.extend(['','## Text similarity and potential unattributed overlap','',
                  f"Similarity within checked sources: {analysis['overlap_percentage']:.2f}%",
                  f"Potentially unattributed overlap within checked sources: {analysis['unattributed_overlap_percentage']:.2f}%",
                  f"Quoted and source-cited overlap: {analysis['quoted_cited_overlap_percentage']:.2f}%",
                  f"Unique overlapping tokens: {analysis['overlapping_words']}; denominator (examined body tokens): {analysis['analyzed_body_words']}",
                  f"Submitted text examined: {analysis['analyzed_body_words']} / {analysis['total_body_words']} extracted body tokens",
                  f"Semantic comparison: {analysis['semantic_status']}",'','## Search coverage',''])
    for row in analysis['coverage']:
        lines.append(f"- {row['name']} — {row['origin']} — {row['status']} — words checked: {row['words_checked']} — {safe_link(row.get('link',''))}")
    for m in analysis['matches']:
        lines.extend(['',f"### {m['match_type']} — {m['source_name']}",
                      f"Source origin: {m['source_origin']}; associated references: {m['source_reference_ids']}",
                      f"Source link: {safe_link(m['source_link']) or 'User-uploaded source; no public link'}",
                      f"Submitted pages: {m['submitted_pages']}; source pages: {m['source_pages'] or 'Not available (full-text XML)'}",
                      f"Attribution: {m['attribution']}; observed citations: {m['citation_text']}",
                      f"Similarity ({'cosine; not overlap' if m['match_type']=='semantic similarity' else 'token sequence ratio'}): {m['similarity']}",
                      'Submitted passage:',literal(m['submitted_passage']),'Matching source passage:',literal(m['source_passage'])])
    lines.extend(['','## AI-writing assessment','',ai['status'],ai.get('explanation','')])
    if ai.get('provider'):
        import json
        lines.extend([f"Provider: {ai['provider']}; requested model/version: {ai['requested_version']}; reported version: {ai['reported_version']}",
                      f"Document classification: {ai['classification']}",
                      'Class probabilities (not percentage of text written by AI):',literal(json.dumps(ai['class_probabilities'])),
                      'Passage-level provider results:',literal(json.dumps(ai.get('sentences',[]),ensure_ascii=False)),
                      f"Provider documentation: {ai['documentation']}",f"Detector text characters sent: {ai['characters_sent']}"])
        lines.extend([f"Detector verdict: {ai.get('verdict',ai['classification'])}; provider confidence category: {ai.get('confidence_category','Not returned')}",
                      f"AI-generation probability (AI_ONLY confidence, not fraction written by AI): {ai.get('ai_generation_probability','Not returned')}",
                      'Analysis coverage (bibliography excluded):',literal(json.dumps(ai.get('coverage',{}))),
                      f"Percentage of analyzed text flagged as potentially AI-generated: {str(ai['flagged_percentage'])+'%' if ai.get('flagged_percentage') is not None else 'Unavailable'}",
                      f"Flagged words: {ai.get('flagged_words','Unavailable')}; analyzed words: {ai.get('analyzed_words','Unavailable')}",
                      ai.get('threshold','No supported sentence flags'),ai.get('percentage_method',''),ai.get('passage_note',''),
                      f"Sentence-flag coverage: {ai.get('passage_covered_words','Unavailable')} words; unaligned passages: {ai.get('unaligned_passages','Unavailable')}",
                      'For partial coverage, the verdict and probabilities apply only to the submitted excerpt. Remaining text was not assessed.'])

    else:
        lines.append(literal(ai.get('observations',{})))
    if analysis.get('consents'):
        lines.extend(['','## External-service consent for this run',''])
        for key,value in analysis['consents'].items():
            lines.append(f'- {key}: {value}')
    if ai.get('external_submission_attempted'):
        lines.append(f"GPTZero body submission attempted: {ai['characters_attempted']} characters; receipt/completion is not guaranteed on a failed request.")
    lines.extend(['','## Methodology','',METHOD,analysis['candidate_limits'],
                  '', '## Extraction notes','', *paper.extraction_notes,
                  '', '## Limitations','',LIMITATIONS,
                  'Detector false positives/negatives, editing, translation, genre, language and distribution shifts limit authorship inferences. No result establishes academic misconduct.'])
    return '\n'.join(lines)
