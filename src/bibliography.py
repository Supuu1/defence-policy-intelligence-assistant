"""Conservative structural bibliography parsing; unknown fields stay empty."""
from collections import Counter
import re

YEAR=re.compile(r'\b((?:18|19|20)\d{2}[a-z]?)\b')
HEADING=re.compile(r'^\s*(?:(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+)?(?:references(?: and notes)?|bibliography|works cited|literature cited|reference list)\s*$',re.I)
NUMBER=re.compile(r'^\s*(?:\[(\d+)\]|(\d+)[.)])\s+(.+)')
AUTHOR=re.compile(r"^(?:[A-ZÀ-ÖØ-Þ][\w’'\-]+,\s*(?:[A-Z]\.|[A-Z][a-z]+)|(?:[A-Z]\.\s*)+[A-Z][\w’'\-]+)")
BARE_AUTHOR=re.compile(r"^[A-ZÀ-ÖØ-Þ][\w’'\-]+(?:\s+[a-z]+)?\s+[A-Z](?:\s+[A-Z])?(?:\s+(?:and|&|[A-Z][\w’'\-]+)|\s+(?:18|19|20)\d{2})")
BODY=re.compile(r'^(?:(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+)?(?:introduction|classical information theory|conclusions?|abstract|acknowledg(?:e)?ments?|contents|appendi(?:x|ces)(?:\s+[A-Z0-9]+)?|supplementary (?:material|information))\s*$',re.I)


def valid_authors(value):
    surname=r"[A-ZÀ-ÖØ-Þ][\w’'\-]+"
    initials=r'(?:[A-Z]\.\s*){1,4}'
    given=r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'
    person=rf"(?:{surname},\s*(?:{initials}|{given})|{initials}{surname}|{surname}\s+(?:[A-Z](?:\s+|$)){{1,4}})"
    return bool(re.fullmatch(rf'{person}(?:\s*(?:,|;|and|&)\s*{person})*\s*',value))


def clean_lines(pages):
    rows=[];edges=Counter()
    for index,page in enumerate(pages):
        lines=page.get('text','').splitlines()
        nonempty=[i for i,line in enumerate(lines) if line.strip()]
        edge=set(nonempty[:2]+nonempty[-2:])
        seen=set()
        for i,line in enumerate(lines):
            normalized=re.sub(r'\d+','#',line.strip()).casefold()
            candidate=len(nonempty)>=4 and i in edge and not HEADING.fullmatch(line) and not NUMBER.match(line) and not AUTHOR.match(line) and not BARE_AUTHOR.match(line)
            if candidate and normalized and normalized not in seen:
                edges[normalized]+=1;seen.add(normalized)
            rows.append((line,page.get('page_number'),index,i,candidate,normalized))
    result=[];removed=0
    for line,page,_,_,edge,key in rows:
        if (line.strip()==str(page) and re.fullmatch(r'\s*\d+\s*',line)) or (edge and key!='#' and edges[key]>=2):
            removed+=1;continue
        result.append((line,page))
    return result,removed


def bibliography_layout(pages):
    """Exact heading plus plausible entry structure; never a year-only boundary."""
    rows,removed=clean_lines(pages)
    headings=[]
    for i,(line,_) in enumerate(rows):
        if not HEADING.fullmatch(line): continue
        preceding=[text.strip().casefold() for text,_ in rows[max(0,i-12):i]]
        if i<len(rows)//2 and any(text in ('contents','table of contents') for text in preceding):continue
        following=[text.strip() for text,_ in rows[i+1:i+16] if text.strip()]
        if any(NUMBER.match(text) or AUTHOR.match(text) or BARE_AUTHOR.match(text) for text in following):
            headings.append(i)
    # An earlier table-of-contents occurrence cannot supersede the actual section.
    start=headings[-1] if headings else None
    headingless=None
    if start is None:
        # Dense runs of surname+initials+year entries in the latter half, not a
        # year pattern or a body mention. Useful for headingless physics reviews.
        for i,(line,_) in enumerate(rows):
            if i<len(rows)//2 or not BARE_AUTHOR.match(line.strip()):continue
            following=[text.strip() for text,_ in rows[i:i+64] if text.strip()]
            candidates=[text for text in following if BARE_AUTHOR.match(text) and YEAR.search(text) and valid_authors(text[:YEAR.search(text).start()].strip())]
            if len(candidates)>=5:
                headingless=i;break
    body=[];refs=[];inside=False
    for i,(line,page) in enumerate(rows):
        if i==start:
            inside=True;continue
        if i==headingless:inside=True
        if inside and HEADING.fullmatch(line): continue
        if inside and (BODY.match(line.strip()) or re.match(r'^\s*(?:Fig(?:ure)?\.?\s*\d+|Figure captions)\b',line,re.I)):
            inside=False
        (refs if inside else body).append((line+'\n',page))
    return body,refs,start is not None or headingless is not None,removed


def parse_entries(rows):
    entries=[];current=[];pages=[];label=None
    def flush():
        if not current:return
        raw=' '.join(current).strip()
        year_matches=list(YEAR.finditer(raw))
        # Author-year style puts year before title; numbered physics styles often put it last.
        ym=year_matches[0] if year_matches else None
        year=ym.group(1) if ym else ''
        authors='';title='';surname=''
        author_year=re.match(r"^(.{2,220}?)\s*\(((?:18|19|20)\d{2}[a-z]?)\)\s*[.,:]?\s*(.+)$",raw)
        if author_year and AUTHOR.match(raw) and not re.search('[“”\"]',raw[:author_year.start(2)]):
            authors,year,tail=author_year.groups()
            title=re.split(r'\.\s|https?://|doi:',tail,maxsplit=1,flags=re.I)[0].strip(' .,:;')
        else:
            bare=re.match(r'^(.{2,220}?)\s+((?:18|19|20)\d{2}[a-z]?)\s+(.+)$',raw) if BARE_AUTHOR.match(raw) else None
            if bare:
                authors,year,tail=bare.groups()
                title=re.split(r',\s*(?:Phys|Nature|Science|Proc|Rev|J\.|Rep)|\.\s|https?://|doi:',tail,maxsplit=1,flags=re.I)[0].strip(' .,:;')
            # Author list terminated by a sentence, or a quoted title in physics references.
            quoted=re.match(r'^(.{2,220}?)\s*[,“"]\s*[“"]([^”"]+)[”"]',raw)
            if quoted and not bare:
                authors,title=quoted.groups()
            elif AUTHOR.match(raw) and not bare:
                boundary=re.search(r'(?<!\b[A-Z])\.\s+(?=[A-Z“"])',raw)
                if boundary:
                    authors=raw[:boundary.start()]
                    tail=raw[boundary.end():]
                    title=re.split(r'\.\s|https?://|doi:',tail,maxsplit=1,flags=re.I)[0].strip(' .,:;')
        authors=authors.strip(' ,;')
        # Require an actual author-shaped prefix, bounded fields, title, and a year.
        valid=bool(valid_authors(authors) and 2<=len(authors)<=220 and 3<=len(title)<=300 and year)
        if valid:
            initial_first=re.match(r'^(?:[A-Z]\.\s*)+([A-Z][\w’\'\-]+)',authors)
            surname=(initial_first.group(1) if initial_first else authors.split(',')[0] if not BARE_AUTHOR.match(authors+' '+year) else authors.split()[0]).casefold()
        entries.append(dict(label=label,raw=raw,pages=list(dict.fromkeys(pages)),
                            authors=authors if valid else '',title=title if valid else '',
                            year=year if valid else '',surname=surname if valid else '',
                            parse_status='Parsed reference' if valid else 'Unparsed reference',
                            lookup_eligible=valid,excerpt=raw[:180]))
    for line,page in rows:
        line=line.strip()
        if not line:continue
        if BODY.match(line):
            flush();current=[];pages=[];label=None;continue
        numbered=NUMBER.match(line)
        if not numbered and re.match(r'^(?:This paper|This section|We (?:show|present|discuss)|The |In this )',line) and len(line.split())>25:
            flush();current=[];pages=[];label=None;continue
        author_start=AUTHOR.match(line) or BARE_AUTHOR.match(line)
        if numbered or (author_start and current and YEAR.search(' '.join(current))):
            flush();current=[];pages=[];label=None
        if numbered:
            label=int(numbered.group(1) or numbered.group(2));line=numbered.group(3)
        # Ordinary prose outside an entry is rejected, not sent to metadata services.
        if not current and not numbered and not author_start:
            if len(line.split())>25 or re.match(r'^(?:This|The|We|In|It|These|Our)\b',line):continue
            if len(line)>3:
                current=[line];pages=[page];flush();current=[];pages=[]
            continue
        current.append(line);pages.append(page)
    flush()
    return entries
