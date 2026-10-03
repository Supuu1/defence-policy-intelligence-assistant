"""Free CPU-only pretrained classifier. Paper text never leaves this process."""
from pathlib import Path
import math
import os
import threading
from bisect import bisect_left
from contextlib import nullcontext
import streamlit as st
from src.research_integrity import tokens, pages_at

MODEL='MayZhou/e5-small-lora-ai-generated-detector'
REVISION='483fc4969592dc20e00e5130e7187b5dd25dbcc7'
MODEL_CARD='https://huggingface.co/'+MODEL
LIMITATIONS=('Experimental English classifier trained on RAID and GPT-4o-mini rewritten tweets. '
    'Scores are uncalibrated softmax outputs for label 1 (AI-generated), not authorship probabilities. '
    'False positives and false negatives are possible, especially for academic writing, edited text '
    'and generators outside its training data. Text alone cannot establish authorship or misconduct.')
METHOD=('Analyzed text flagged as potentially AI-generated: union of word-token positions in '
    'successfully classified passages with label-1 score >= configured threshold / union of all '
    'successfully analyzed word-token positions × 100. Bibliography excluded; duplicate/overlapping '
    'positions count once. This is not the actual percentage of AI used. No document probability '
    'is computed by averaging passage scores.')


def local_defaults():
    """Optional server defaults; users can change both controls in the UI."""
    from dotenv import load_dotenv
    load_dotenv()
    result=dict(threshold=.8,max_passages=200,warnings=[])
    for key,name,convert,default in [('LOCAL_AI_THRESHOLD','threshold',float,.8),('LOCAL_AI_MAX_PASSAGES','max_passages',int,200)]:
        raw=os.getenv(key)
        if raw is None:
            try:raw=st.secrets.get(key)
            except Exception:raw=None
        if raw is None:continue
        try:
            value=convert(raw)
            if name=='threshold' and (not math.isfinite(value) or not .01<=value<=.99):raise ValueError()
            if name=='max_passages' and not 1<=value<=1000:raise ValueError()
            result[name]=value
        except (ValueError,TypeError):result['warnings'].append(f'Invalid {key}; using documented default {default}. Correct server settings or adjust the UI control.')
    return result


def memory_failure(error):
    return isinstance(error,MemoryError) or (isinstance(error,(RuntimeError,OSError)) and any(phrase in str(error).lower() for phrase in ('out of memory','cannot allocate memory',"can't allocate memory")))


class LocalDetectorError(RuntimeError):pass


@st.cache_resource(show_spinner=False)
def _load_detector(revision):
    """Shared model/lock only, never shared paper text or inference results."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    cache=Path(__file__).resolve().parents[1]/'.cache'/'local-detector'
    options=dict(revision=revision,cache_dir=str(cache),trust_remote_code=False)
    # Public model only. No token or private paper data is passed to Hugging Face.
    def load(offline):
        tokenizer=AutoTokenizer.from_pretrained(MODEL,use_fast=True,token=False,local_files_only=offline,**options)
        model=AutoModelForSequenceClassification.from_pretrained(MODEL,use_safetensors=True,token=False,local_files_only=offline,**options)
        return tokenizer,model
    try:
        tokenizer,model=load(True)
    except OSError as error:
        if memory_failure(error):raise MemoryError() from None
        tokenizer,model=load(False)
    model.to('cpu').eval()
    torch.set_num_threads(2)
    if model.config.num_labels!=2 or model.config.max_position_embeddings!=512:
        raise LocalDetectorError('Unsupported model configuration')
    return tokenizer,model,threading.Lock()


def load_detector():
    # Include the pinned revision in the resource cache key, not paper contents.
    return _load_detector(REVISION)


def split_passages(text,tokenizer,limit=512):
    """Nonoverlapping windows, measured with the actual tokenizer, never truncating."""
    capacity=limit-tokenizer.num_special_tokens_to_add(pair=False)
    if capacity<=0:raise LocalDetectorError('Unsupported token limit')
    encoded=tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True,verbose=False)
    offsets=encoded['offset_mapping']
    if any(not 0<=a<b<=len(text) for a,b in offsets):
        raise LocalDetectorError('Unsupported tokenizer offsets')
    words=tokens(text);starts=[a for _,a,_ in words]
    chunks=[];skipped=[];i=0
    while i<len(offsets):
        stop=min(i+capacity,len(offsets));start=offsets[i][0];end=offsets[stop-1][1]
        # Back off if this token window splits a word. A word exceeding the entire
        # token budget is skipped explicitly rather than truncated or split-scored.
        word_index=bisect_left(starts,end)-1
        if word_index>=0 and words[word_index][1]<end<words[word_index][2]:
            end=words[word_index][1]
            while stop>i and offsets[stop-1][1]>end:stop-=1
        if stop==i:
            word_end=words[word_index][2] if word_index>=0 else offsets[i][1]
            skipped.append(dict(start=start,end=word_end,reason='Word exceeds model token budget'))
            while i<len(offsets) and offsets[i][0]<word_end:i+=1
            continue
        chunk=text[start:end]
        check=tokenizer(chunk,add_special_tokens=True,truncation=False,verbose=False)['input_ids']
        if len(check)>limit:raise LocalDetectorError('Token splitting validation failed')
        if chunk.strip():chunks.append(dict(text=chunk,start=start,end=end,model_tokens=len(check)))
        i=stop
    return chunks,skipped


def score_passage(text,resource):
    import torch
    tokenizer,model,lock=resource
    with lock,torch.inference_mode():
        encoded=tokenizer(text,return_tensors='pt',truncation=False)
        if encoded['input_ids'].shape[-1]>512:raise LocalDetectorError('Token limit exceeded')
        logits=model(**encoded).logits
        value=float(torch.softmax(logits,dim=-1)[0,1].item())
    if not math.isfinite(value) or not 0<=value<=1:raise LocalDetectorError('Invalid classifier output')
    return value


def assess_local_writing(paper,enabled=True,threshold=.8,language='English',max_passages=200,resource=None):
    total=tokens(paper.body)
    base=dict(status='Experimental local AI-writing assessment',local_detector=True,model=MODEL,version=REVISION,
              license='MIT (model-card declaration)',documentation=MODEL_CARD,explanation=LIMITATIONS,
              assessment='Uncertain — no reliable document-level authorship verdict',sentences=[],threshold=threshold,
              method=METHOD,flagged_percentage=None,flagged_words=0,analyzed_words=0,skipped=[],
              coverage=dict(analyzed_words=0,total_body_words=len(total),percent=0,pages=[],skipped_words=len(total),
                            bibliography_entries_excluded=len(paper.bibliography),bibliography_words_excluded=sum(len(tokens(r['raw'])) for r in paper.bibliography),bibliography_found=paper.bibliography_found))
    def unavailable(message):
        base['error']=message
        if total and not base['skipped']:
            base['skipped']=[dict(start=0,end=len(paper.body),reason=message)]
        return base
    if not enabled:return unavailable('Local detector not run. Enable the local assessment; no API key is needed.')
    if language!='English':return unavailable('Unsupported language: this model is supported for English only. No scores were inferred.')
    if not total:return unavailable('No readable non-bibliography words are available.')
    if not isinstance(threshold,(int,float)) or isinstance(threshold,bool) or not math.isfinite(threshold) or not 0<threshold<1:
        return unavailable('Unsupported threshold: choose a finite number strictly between 0 and 1.')
    if not isinstance(max_passages,int) or not 1<=max_passages<=1000:return unavailable('Passage limit must be an integer from 1 to 1000.')
    analyzed=set();flagged=set()
    try:
        resource=load_detector() if resource is None else resource
        with resource[2] if resource[2] is not None else nullcontext():
            chunks,skipped=split_passages(paper.body,resource[0])
        base['skipped']=skipped
        starts=[a for _,a,_ in total]
        for index,chunk in enumerate(chunks):
            if index>=max_passages:
                base['skipped'].append(dict(start=chunk['start'],end=chunk['end'],reason='Configured passage-count limit'))
                continue
            left=bisect_left(starts,chunk['start']);right=bisect_left(starts,chunk['end'])
            indices={i for i in range(left,right) if total[i][2]<=chunk['end']}
            try:
                score=score_passage(chunk['text'],resource)
                if isinstance(score,bool) or not isinstance(score,(float,int)) or not math.isfinite(score) or not 0<=score<=1:
                    raise LocalDetectorError('Invalid classifier output')
            except MemoryError:
                base['error']='Insufficient memory during CPU inference. Reduce passage load or increase hosting memory.'
                base['skipped'].append(dict(start=chunk['start'],end=len(paper.body),reason='Memory failure; remaining text not assessed'))
                break
            except Exception as error:
                if memory_failure(error):
                    base['error']='Insufficient memory during CPU inference. Increase hosting memory; no score was inferred for remaining text.'
                    base['skipped'].append(dict(start=chunk['start'],end=len(paper.body),reason='Memory failure; remaining text not assessed'))
                    break
                base['skipped'].append(dict(start=chunk['start'],end=chunk['end'],reason='Inference failure; no score inferred'))
                continue
            analyzed.update(indices)
            if score>=threshold:flagged.update(indices)
            base['sentences'].append(dict(chunk,score=score,flagged=score>=threshold,pages=pages_at(paper,chunk['start'],chunk['end']),
                                           meaning='Uncalibrated label-1 classifier score; not a document probability'))
        base['analyzed_words']=len(analyzed);base['flagged_words']=len(flagged)
        if analyzed:base['flagged_percentage']=100*len(flagged)/len(analyzed)
        base['coverage'].update(analyzed_words=len(analyzed),percent=100*len(analyzed)/len(total),
                                skipped_words=len(total)-len(analyzed),
                                pages=list(dict.fromkeys(p for item in base['sentences'] for p in item['pages'])))
        if not analyzed:base['error']='No passages were successfully classified. Check model availability, resources and readable English text.'
        if len(analyzed)<len(total):base['assessment']='Uncertain — partial coverage; remaining words were not assessed'
        return base
    except MemoryError:
        return unavailable('Insufficient memory to load the CPU detector. Increase hosting memory; no substitute detector was used.')
    except (OSError,ConnectionError) as error:
        if memory_failure(error):return unavailable('Insufficient memory to load the CPU detector. Increase hosting memory; no substitute detector was used.')
        return unavailable('Model download/cache unavailable. Allow outbound HTTPS to Hugging Face for initial model files, check disk space, or prepopulate .cache/local-detector.')
    except ImportError:
        return unavailable('Detector dependency missing. Install the repository requirements and reboot the app.')
    except Exception as error:
        if memory_failure(error):return unavailable('Insufficient memory to load the CPU detector. Increase hosting memory; no substitute detector was used.')
        return unavailable('Local model loading/tokenization failed. Check pinned model files and Transformers compatibility; no score was inferred.')
