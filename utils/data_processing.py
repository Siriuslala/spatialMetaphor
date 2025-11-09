import jsonlines
from huggingface_hub import snapshot_download
from langdetect import detect
import pandas as pd
from tqdm import tqdm
# import nltk
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
# python -m spacy download en_core_web_sm
import zstandard as zstd
import re
from tqdm import tqdm
import subprocess
from typing import List, Dict, Set
import requests
from tqdm import tqdm

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
ROOT_DIR = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
DATA_DIR = Path(os.getenv('DATA_DIR'))
DATA_DIR_PUBLIC = Path(os.getenv('DATA_DIR_PUBLIC'))
WORK_DIR = Path(os.getenv('WORK_DIR'))
HF_PILE_KEY=os.getenv('HUGGINGFACE_PILE_KEY')
ZHIPU_API_KEY=os.getenv('ZHIPU_API_KEY')
sys.path.append(ROOT_DIR.as_posix())

from data.const import *
from utils.model_heplers import load_model_and_tokenizer, batch_infer


nlp = spacy.load("en_core_web_sm")

concept_name_to_keywords = {
    "en": {
        "emotion_positive": EMOTION_POSITIVE_KEWORDS_EN,
        "emotion_negative": EMOTION_NEGATIVE_KEWORDS_EN,
        "orientation_positive": ORIENTATION_POSTTIVE_KEYWORDS_EN,
        "orientation_negative": ORIENTATION_NEGATIVE_KEYWORDS_EN
    }
}


def get_huggingface_dataset(dataset_name, hf_key=None):
    # command = "huggingface-cli download --repo-type dataset --token 'hf-...' --resume-download 数据集名称 --cache-dir '' --local-dir-use-symlinks False"
    snapshot_download(
        repo_id=dataset_name, 
        repo_type="dataset",
        # cache_dir="/raid_sdd/lyy/dataset",
        # local_dir_use_symlinks=False, 
        resume_download=True,
        token=hf_key,
    )

def decompress_zst_file(input_path: str, output_path: str):
    print(f"Start decompressing file: {input_path}")
    print(f"Target output file: {output_path}")

    dctx = zstd.ZstdDecompressor()
    
    try:
        with open(input_path, 'rb') as ifh:
            with open(output_path, 'wb') as ofh:
                dctx.copy_stream(ifh, ofh)
        print(f"\n✅ Completed decompression! File saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ Error during decompression: {e}")

def decompress_wiki(input_path, output_path=None):
    # curl -C - -o [target_path] https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2      
    # python 3.10
    # pip install Wikiextractor
    # python -m wikiextractor.WikiExtractor -b 1024M -o [output_dir] [input_path]
    output_path = DATA_DIR / "wiki_en"
    command = f"python -m wikiextractor.WikiExtractor --json -b 1024M -o {output_path.as_posix()} {input_path}"
    subprocess.run(command, shell=True, check=True)

def process_data_wiki_stage1(input_dir, output_dir):
    """ 
    Stage1: remove formating tags
    """
    # under input dir: wiki_01, ..., wiki_17
    files = os.listdir(input_dir)
    for file_name in tqdm(files, desc="Processing wiki files"):
        file_path = input_dir / file_name
        outfile_path = output_dir / f"{file_name}.txt"
        with open(outfile_path, 'w', encoding='utf-8') as f:
            with open(file_path, 'r', encoding='utf-8') as f0:
                articles = f0.readlines()
                article_list = []
                for idx in range(len(articles)):
                    article = articles[idx]
                    if article == '\n':
                        del article
                    else:
                        article = article.strip()
                        if not article.endswith('.') and not article.endswith('>'):
                            article = article+'. '
                        article_list.append(article)
                outarticles = ' '.join(article_list)  # concat all articles together
                re_out = re.findall(r'<doc.*?>(.*?)</doc>', outarticles)
                f.writelines(re_out)    
                # break

def process_data_wiki_stage2(input_dir, output_dir, max_data_num=50000):
    """
    Stage2: split into sentences
    """
    files = os.listdir(input_dir)
    for file_name in tqdm(files, desc="Processing wiki files"):
        file_path = input_dir / file_name
        outfile_path = output_dir / file_name
        with open(outfile_path, 'w', encoding='utf-8') as f:
            with open(file_path, 'r', encoding='utf-8') as f0:
                wiki_lines = f0.read().replace('. ', '.\n')
                wiki_lines = wiki_lines.splitlines()
                print(len(wiki_lines))
            cnt = 0
            for line in wiki_lines:
                line = line.strip()
                f.writelines(line + '\n')
                cnt += 1
                if max_data_num is not None and cnt >= max_data_num:
                    break

def process_data_wiki_stage3(input_dir, output_dir, keyword="happy"):
    """
    Stage3: filter out sentences containing keywords
    """
    happy_regex = re.compile(r'(?<![a-zA-Z0-9_-])happy(?![a-zA-Z0-9_-])')
    sad_regex = re.compile(r'(?<![a-zA-Z0-9_-])sad(?![a-zA-Z0-9_-])')
    # up_regex = re.compile(r'\bup\b')
    # down_regex = re.compile(r'\bdown\b')
    up_regex = re.compile(r'(?<![a-zA-Z0-9_-])up(?![a-zA-Z0-9_-])')
    down_regex = re.compile(r'(?<![a-zA-Z0-9_-])down(?![a-zA-Z0-9_-])')
    
    files = os.listdir(input_dir)
    output_path = output_dir / f"wiki_en_{keyword}.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        for file_name in tqdm(files, desc="Processing wiki files"):
            file_path = input_dir / file_name
            with open(file_path, 'r', encoding='utf-8') as f0:
                lines = f0.readlines()
                for line in tqdm(lines, desc="Processing lines"):
                    line = line.strip()
                    if keyword in ["up", "down"]:
                        has_up = up_regex.findall(line)
                        has_down = down_regex.findall(line)
                        condition1 = keyword == "up" and has_up and not has_down and len(has_up) == 1
                        condition2 = keyword == "down" and has_down and not has_up and len(has_down) == 1
                        if not condition1 and not condition2:
                            continue
                        doc = nlp(line)
                        words = [token.text for token in doc]
                        if not check_orientational_property(doc, keyword):
                            continue
                        # tokens = word_tokenize(line)
                        # words = [word for word in tokens if word.isalpha()]
                        if not check_length(words, min_len=5, max_len=40):  # too short or too long
                            continue
                    else:
                        has_happy = happy_regex.findall(line)
                        has_sad = sad_regex.findall(line)
                        condition1 = keyword == "happy" and has_happy and not has_sad and len(has_happy) == 1
                        condition2 = keyword == "sad" and has_sad and not has_happy and len(has_sad) == 1
                        if not condition1 and not condition2:
                            continue
                        tokens = word_tokenize(line)
                        words = [word for word in tokens if word.isalpha()]
                        if not check_length(words, min_len=5, max_len=40):  # too short or too long
                            continue
                    f.writelines(line + '\n')

def check_length(words, min_len=5, max_len=40):
    if len(words) <= min_len or len(words) >= max_len:
        return False
    return True

ABSTRACT_PHRASAL_VERBS: Dict[str, Set[str]] = {
    "up": {
        "give", "grow", "show", "make", "set", "use", "eat",
        "cheer", "end", "back", "bring", "break", "speak",
        "call", "clean", "heat", "dream", "think", "wake", "hurry",
        "start", "sign", "mess", "open", "read", "write", "dress",
        "line", "team", "fix", "cut", "come", "catch", 
        "drink", "ring", "stay", "shut", "warm", "dry", "burn", "finish",
        "blow", "screw", "light", "tidy", "pile", "fill", "mix",
        "link", "block", "check", "clear", "cry", "build", "drive",
        "walk", "keep", "add", "count", "collect", "gather", "round",
        "join", "number", "tear", "divide", "separate", "lock", "fasten",
        "stick", "nail", "tie", "wrap", "split", "beat", "take", 
        "crop", "wall", "follow", "sweep", "lead", "dig",
        "draw", "meet", "pluck", "pair", "hang", "put", "hook", 
        "fold", "swallow", "close", "winch", "lay", "drum",
        "grass", "rest", "fuck",
    },
    "down": {
        "break", "shut", "calm", "slow", "write", "note",
        "let", "cut", "back", "settle", "hunt", "track", 
        "close", "put", "run", "burn", "stand"
    },  # write, note, take 等词跟 down 搭配使用时，”向下“的意味并不明显
    # step down(下台)
    
    # "down": {
    #     "break", "shut", "calm", "turn", "slow", "write", "note",
    #     "let", "cut", "back", "settle", "hunt", "track", "get",
    #     "close", "put", "take", "lie", "run", "fall", "come"
    # }
}

ABSTRACT_SUBJECTS: Set[str] = {
    "price", "rate", "level", "temperature", "economy", "stock", "market",
    "hope", "feeling", "mood", "number", "total", "percentage", "option",
    "responsibility", "pressure", "volume", "cost", "value", "speed",
    "heat", "system", "website", "server", "internet", "power"
}  # 停用

def check_orientational_property(sentence_doc, keyword):
    reject_ner_types = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "PERSON"}
    for token in sentence_doc:
        if token.text == keyword:
            # 规则 1: 检查介词 (prep)
            # e.g., "up the mountain", "up to 9 pages"
            if token.dep_ == "prep":
                # 1.1 检查已知的非空间习语 (e.g., "up to", "down to", "up until")
                if token.i + 1 < len(sentence_doc):
                    next_token = sentence_doc[token.i + 1]
                    phrase = f"{token.lemma_} {next_token.lemma_}"
                    if phrase in ("up to", "down to", "up until"):
                        return False  # "up to 9", "down to the detail", "up until 1973", "up to you"
                    
                # 1.2 检查介词宾语 (pobj)
                pobj = next((child for child in token.children if child.dep_ == "pobj"), None)
                if pobj:
                    # 如果宾语是数字、百分比等，判定为非空间
                    if pobj.pos_ == "NUM" or pobj.like_num or pobj.ent_type_ in reject_ner_types:
                        return False # "up 9 pages"
                
                # 1.3 如果通过了上述检查，作为介词的 "up/down" 倾向于是空间性的
                # e.g., "up the mountain", "down the stairs"
                return True
                
            # 规则 2: 检查小品词 (prt) - 短语动词
            # e.g., "flew up", "give up"
            if token.dep_ == "prt":  # e.g. give up, write down
                verb = token.head
                if verb.pos_ != "VERB":
                    return False

                # 2.0 case by case
                # "look up"
                if verb.lemma_ == "look" and token.lemma_ == "up":
                    # 检查 'look' 是否有直接宾语 (dobj)
                    # 如果有 (e.g., "looked up [the word] in the dictionary"), 则判定为非空间性
                    if any(child.dep_ == 'dobj' for child in verb.children):
                        return False
                    return True
                
                # "turn up/down"
                if verb.lemma_ == "turn":
                    dobj = next((child for child in verb.children if child.dep_ == 'dobj'), None)
                    if dobj:
                        return True  # 后续再筛
                        # 检查宾语是否抽象 (e.g., "turn up/down [the volume/offer]")
                        if dobj.lemma_ in ABSTRACT_SUBJECTS or dobj.lemma_ in {"offer", "request", "invitation"}:
                            return False
                        # 宾语具体 (e.g., "turn up [the card]", "turn down [the blinds]") -> 空间
                        return True
                    else:
                        if token.lemma_ == "up":
                            return False # 抽象 (出现, "He turned up")
                        else:
                            return True # 空间 (e.g., "The covers were turned down")
                        
                # "pull up/down"
                if verb.lemma_ == "pull":
                    dobj = next((child for child in verb.children if child.dep_ == 'dobj'), None)
                    if dobj:
                        return True  # 后续再筛
                        # 检查特殊抽象宾语 (e.g., "pull down [a building]")
                        if token.lemma_ == "down" and dobj.lemma_ in {"building", "house", "structure", "statue"}:
                            return False # 抽象 (拆除)
                        # 宾语具体 (e.g., "pull up [his socks]", "pull down [the blinds]") -> 空间
                        return True
                    else:
                        if token.lemma_ == "up":
                            return False # 抽象 (停车, "The car pulled up")
                        else:
                            return True # 空间 (e.g., "The blinds pull down")
                
                # 4. "get up/down"
                if verb.lemma_ == "get":
                    dobj = next((child for child in verb.children if child.dep_ == 'dobj'), None)
                    if dobj:
                        return False  # e.g., "This gets [me] down" -> 抽象 (沮丧) (get ... up/down 语义较为模糊，pass)
                    # 没有 dobj (e.g., "He got up", "Get down!") -> 空间
                    return True
            
                # 2.1 检查是否为已知的抽象短语动词
                if verb.lemma_ in ABSTRACT_PHRASAL_VERBS[token.lemma_]:
                    return False # "give up", "grow up", etc.
                
                # 2.2 检查动词主语
                # subject = next((child for child in verb.children if child.dep_ in ("nsubj", "nsubjpass")), None)
                # if subject:
                #     if subject.lemma_.lower() in ABSTRACT_SUBJECTS or subject.text.lower() in ABSTRACT_SUBJECTS:
                #         return False # "The price rises up."
                
                # 2.3 如果主语不是抽象的，且不是明确的抽象短语，则判定为空间性
                # e.g., "The bird flew up", "He jumped up"
                return True

            # 规则 3: 检查补语或状语 (acomp, advmod)
            # e.g., "The price is up", "He is up", "He went up"
            if token.dep_ in ("acomp", "advmod"):
                if token.i + 1 < len(sentence_doc):
                    next_token = sentence_doc[token.i + 1]
                    phrase = f"{token.lemma_} {next_token.lemma_}"
                    if phrase in ("up to", "down to", "up until", "up against"):
                        return False
                
                if token.i - 1 >= 0:
                    pre_token = sentence_doc[token.i - 1]
                    if pre_token.text.lower() in ["further"]:
                        return False  # "further up"

                verb = token.head
                # 3.1 检查主语
                # subject = next((child for child in verb.children if child.dep_ in ("nsubj", "nsubjpass")), None)
                # if subject:
                #     if subject.lemma_.lower() in ABSTRACT_SUBJECTS or subject.text.lower() in ABSTRACT_SUBJECTS:
                #         return False # "The price is up."
                
                # 3.2 如果主语是具体的，则判定为空间性 (be + up)
                # e.g., "He is up.", "The sun is up."
                return True

            # 默认：如果 "up/down" 不属于上述任何一种主要空间结构，则拒绝
            return False

def process_data_wiki_stage4(
    input_path: Path, 
    output_dir: Path, 
    model_name="Qwen/Qwen3-8B", 
    device='cuda', 
    batchsize=16,
    max_tokens: int=5,
):
    """
    Stage4: (call model) filter out invalid sentences, the context of which is contradict to the keyword
    For happy/sad:
    1: real happy
    2: fake happy
    """
    file_name = input_path.stem
    output_path = output_dir / f"{file_name}.jsonl"
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)

    data = []
    if any(keyword in file_name for keyword in ["happy", "sad"]):
        prompt_template = PROMPT_FILTER_KEYWORD_HAPPY_EN if "happy" in file_name else PROMPT_FILTER_KEYWORD_SAD_EN
    else:
        prompt_template = PROMPT_FILTER_KEYWORD_UP_EN if "up" in file_name else PROMPT_FILTER_KEYWORD_DOWN_EN
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            sample = prompt_template.format(sentence=line).strip()
            data.append(sample)
    labels = []
    regex_input = r"Sentence: (.*)\nCategory ID:"
    regex_ans = r"\s*(\d)"
    batches = [data[i:i+batchsize] for i in range(0, len(data), batchsize)]
    with jsonlines.open(output_path, "w") as f:
        for batch in tqdm(batches, desc="Processing batches"):
            outputs = batch_infer(model, tokenizer, batch, max_tokens=max_tokens)
            results = []
            for i, o in zip(batch, outputs):
                match_input = re.search(regex_input, i)
                match_ans = re.search(regex_ans, o)
                if match_input:
                    sentence = match_input.group(1)
                else:
                    sentence = ""
                if match_ans:
                    category_id = int(match_ans.group(1))
                else:
                    category_id = 0
                labels.append(category_id)
                results.append({
                    "sentence": sentence,
                    "category_id": category_id,
                })
                # print(f"Sentence: {sentence}\nCategory ID: {category_id}")
                # print("----" * 100)
            f.write_all(results)
            # break

def process_data_wiki_stage4_zhipu(
    input_path: Path, 
    output_dir: Path, 
    model_name: str = "glm-4-flash-250414",
):
    """
    Stage4: (call api) filter out invalid sentences, the context of which is contradict to the keyword
    Call Zhipu API to process the data.
    (Unfinished)
    (Deprecated)
    """
    import time
    from zai import ZhipuAiClient

    file_name = input_path.stem
    output_path = output_dir / f"{file_name}.jsonl"

    client = ZhipuAiClient(api_key=ZHIPU_API_KEY)
    file_object = client.files.create(
        file=open(input_path, "rb"),
        purpose="batch"
    )
    batch = client.batches.create(
        input_file_id=file_object.id,
        endpoint="/v4/chat/completions",
        auto_delete_input_file=True,
        metadata={
            "description": "space metaphor_happy",
            "project": "metaphor"
        }
    )

    while True:
        batch_status = client.batches.retrieve(batch.id)
        print(f"任务状态: {batch_status.status}")
        
        if batch_status.status == "completed":
            print("任务完成！")
            break
        elif batch_status.status in ["failed", "expired", "cancelled"]:
            print(f"任务失败，状态: {batch_status.status}")
            break
    
        time.sleep(30)  # 等待30秒后再次检查
    
    if batch_status.status == "completed":
        result_content = client.files.content(batch_status.output_file_id)
        result_content.write_to_file("batch_results.jsonl")
        print("结果文件下载完成: batch_results.jsonl")
        
        # 如果有错误文件，也可以下载
        if batch_status.error_file_id:
            error_content = client.files.content(batch_status.error_file_id)
            error_content.write_to_file("batch_errors.jsonl")
            print("错误文件下载完成: batch_errors.jsonl")

def test_zhipu_api():
    #  website: https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E5%AF%B9%E8%AF%9D%E8%A1%A5%E5%85%A8
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    payload = {
        "model": "glm-4.5-flash",
        "messages": [
            # {
            #     "role": "system",
            #     "content": "你是一个有用的AI助手。"
            # },
            {
                "role": "user",
                "content": "The bird flew up into the sky. 该句话中的 up 是否表示空间意义上的'向上'？如果是，请回答'是'，否则请回答'否'。你的回答是："
            }
        ],
        "temperature": 1,
        "max_tokens": 65536,
        "stream": False
    }
    headers = {
        "Authorization": ZHIPU_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        ans = response['choices'][0]['message']['content']
        print(ans)
        return ans
    else:
        raise Exception(f"API调用失败: {response.status_code}, {response.text}")

def make_zhipu_infer_file(input_path, model_name: str = "glm-4-flash-250414", max_data_num: int = 32):
    file_name = input_path.stem
    output_path = input_path.parent / f"{file_name}_zhipu.jsonl"

    if any(keyword in file_name for keyword in ["happy", "sad"]):
        system_prompt = "You are an expert at sentiment analysis."
    elif any(keyword in file_name for keyword in ["orientation"]):
        system_prompt = "You are an expert at orientation analysis."
    else:
        system_prompt = "You are a helpful assistant."

    with open(input_path, "r") as f:
        lines = f.readlines()
    with jsonlines.open(output_path, "w") as f:
        for idx, line in enumerate(lines):
            line = line.strip()
            prompt = PROMPT_FILTER_KEYWORD_HAPPY_EN.format(sentence=line).strip()
            sample = {
                "custom_id": f"r-{idx}",
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": {
                    "model": model_name, 
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                }
            }
            f.write(sample)
            if idx >= max_data_num:
                break

def process_data_wiki_stage5(input_path, output_dir, max_data_num = 30000):
    """
    Stage5: select valid sentences from the output of Stage4
    """
    file_name = input_path.name
    output_path = output_dir / file_name
    keyword = re.search(r"wiki_en_(.*)\.jsonl", file_name).group(1)

    with jsonlines.open(input_path, "r") as f:
        lines = list(f)
    cnt = 0
    with jsonlines.open(output_path, "w") as f:
        for line in lines:
            if line["category_id"] == 1:
                new_line = {"sentence": line["sentence"], "keyword": keyword}
                f.write(new_line)
                cnt += 1
                if cnt >= max_data_num:
                    break

def is_english(sentence):
    try:
        language = detect(sentence)
        if language == 'en':
            return True
        else:
            return False
    except:
        return False
    
def process_data_pile(data_path, concept_name="emotion_positive", language="en", data_num=100, raw_data_num=10000):
    data_path = data_path.as_posix()
    dataset = None
    cnt = 0
    if ".parquet" in data_path:
        df = pd.read_parquet(data_path)
        dataset = df.to_dict(orient='records')
    elif ".jsonl" in data_path:
        dataset = []
        with jsonlines.open(data_path, "r") as f:
            for line in f:
                dataset.append(line)
                cnt += 1
                if cnt >= raw_data_num:
                    break
    else:
        raise ValueError(f"Unknown data file format: {data_path}")

    keywords = None
    if concept_name is not None:
        keywords = concept_name_to_keywords[language][concept_name]
        if concept_name == "orientation_positive":
            keywords.remove("up")
    
    raw_pile_data_dir = ROOT_DIR / "data/pile10k"
    raw_pile_data_dir.mkdir(parents=True, exist_ok=True)
    if concept_name is not None:
        raw_pile_data_path = raw_pile_data_dir / f"pile_{concept_name}_{language}_{data_num}.jsonl"
    else:
        raw_pile_data_path = raw_pile_data_dir / f"pile_{language}_{data_num}.jsonl"

    cnt = 0
    quit = False
    with jsonlines.open(raw_pile_data_path, "w") as f:
        for data in dataset:
            if data["meta"]["pile_set_name"] in ["Github", "ArXiv", "PubMed Abstracts", "PubMed Central", "StackExchange", "USPTO Backgrounds", "Pile-CC", "DM Mathematics", "FreeLaw"]:   
                continue
            paragraphs = data["text"].split("\n\n")
            for para in paragraphs:
                para = para.replace("\n", " ")
                lines = sent_tokenize(para)
                for i, line in enumerate(lines):
                    # dirty case
                    if any(word in line for word in ["http", "www.", "png", "jpg", "gif"]):
                        continue

                    tokens = word_tokenize(line)
                    words = [word for word in tokens if word.isalpha()]
                    if len(words) <= 5:  # too short
                        continue
                    
                    # find keyword
                    if keywords is not None:
                        keyword = None
                        for k in keywords:
                            if k in words:
                                keyword = k
                                break
                        if keyword is None:
                            continue
                        
                    # language detection -> sentence completeness -> tense detection 
                    conditions = [is_english(line)]
                    if not all(conditions):
                        continue
                    
                    # normal case
                    cnt += 1
                    if keywords is not None:
                        f.write({"sentence": line, "keyword": keyword, "meta": data["meta"]})
                    else:
                        f.write({"sentence": line, "meta": data["meta"]})
                    if cnt >= data_num:
                        quit = True
                        break
                if quit:
                    break
            if quit:
                break
                        
def make_metaphor_data_with_context_templates(output_dir, language="cn", keyword_num=-1):
    lang = language.upper()
    concept_keywords = {
        "emotion_positive": eval(f"EMOTION_POSITIVE_KEWORDS_{lang}"),
        "emotion_negative": eval(f"EMOTION_NEGATIVE_KEWORDS_{lang}"),
        "orientation_positive": eval(f"ORIENTATION_POSTTIVE_KEYWORDS_{lang}"),
        "orientation_negative": eval(f"ORIENTATION_NEGATIVE_KEYWORDS_{lang}")
    }
    if keyword_num > 0:
        for concept in concept_keywords:
            concept_keywords[concept] = concept_keywords[concept][:keyword_num]
    concept_templates = {
        "emotion_positive": eval(f"EMOTION_TEMPLATES_{lang}"),
        "emotion_negative": eval(f"EMOTION_TEMPLATES_{lang}"),
        "orientation_positive": eval(f"SPATIAL_TEMPLATES_{lang}"),
        "orientation_negative": eval(f"SPATIAL_TEMPLATES_{lang}")
    }
    for concept in concept_keywords:
        templates = concept_templates[concept]
        keywords = concept_keywords[concept]
        output_path = output_dir / f"{concept}.jsonl"
        with jsonlines.open(output_path, mode='w') as f:
            for keyword in keywords:
                for template in templates:
                    sentence = template.replace("[CONCEPT]", keyword)
                    f.write({"sentence": sentence, "keyword": keyword})
    print(f"Metaphor data with context templates saved to {output_dir}")


if __name__ == "__main__":
    pass
    
    # Make metaphor dataset  ========================================================
    # language = "en"  # "cn" 或 "en"
    # output_dir = root_dir / f"data/data_single_context/{language}"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # make_metaphor_data_with_context_templates(output_dir, language, keyword_num=1)

    # Get huggingface dataset  ========================================================
    # dataset_name = None  # "monology/pile-uncopyrighted"  "NeelNanda/pile-10k"
    # hf_key = None
    # get_huggingface_dataset(dataset_name, hf_key=hf_key)

    # Uzip .zst file  ========================================================
    # zst_file_path = data_dir_public / "datasets--monology--pile-uncopyrighted/snapshots/3be90335b66f24456a5d6659d9c8d208c0357119/train/00.jsonl.zst"
    # zst_file_path = str(zst_file_path)
    # output_file_path = zst_file_path.removesuffix('.zst')
    # decompress_zst_file(zst_file_path, output_file_path)

    # Decompress .bz2 file  ========================================================
    # bz2_file_path = data_dir / "wiki_en.xml.bz2"
    # decompress_wiki(bz2_file_path.as_posix())

    # Process pile dataset  ========================================================
    # data_path = data_dir_public / "datasets--NeelNanda--pile-10k/snapshots/127bfedcd5047750df5ccf3a12979a47bfa0bafa/data/train-00000-of-00001-4746b8785c874cc7.parquet"
    # data_path = data_dir_public / "datasets--monology--pile-uncopyrighted/snapshots/3be90335b66f24456a5d6659d9c8d208c0357119/train/00.jsonl"
    # process_data_pile(data_path, concept_name="emotion_positive", language="en", data_num=1000, raw_data_num=1000000)
    # process_data_pile(data_path, concept_name=None, language="en", data_num=1000, raw_data_num=1000000)


    # with jsonlines.open(data_path, "r") as f:
    #     for line in f:
    #         if "delighted with almost everything he saw." in line["text"]:
    #             with jsonlines.open("./ignore.txt", "w") as f:
    #                 f.write(line)
    #             break

    # Process wiki dataset  ========================================================
    # Stage1: remove formating tags
    # input_dir = DATA_DIR / "wiki_en/AA"
    # output_dir = DATA_DIR / "wiki_en/AA_processed_0"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # process_data_wiki_stage1(input_dir, output_dir, max_data_num=None)

    # Stage2: split into sentences
    # input_dir = DATA_DIR / "wiki_en/AA_processed_0"
    # output_dir = DATA_DIR / "wiki_en/AA_processed_1"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # process_data_wiki_stage2(input_dir, output_dir, max_data_num=None)

    # Stage3: filter out sentences containing keywords
    # input_dir = DATA_DIR / "wiki_en/AA_processed_1"
    # output_dir = DATA_DIR / "wiki_en/keyword_sentences"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # process_data_wiki_stage3(input_dir, output_dir, keyword="down")

    # Test nlp  ========================================================
    x = "The eyebrows may be up-slanting or outward-slanting."
    y = "An up-rated version was tested as the Piaggio P.XXII."
    def test_tokenize(sentence):
        doc = nlp(sentence)
        words = word_tokenize(sentence)
        print("spacy:", [token.text for token in doc])
        print("nltk:", words)
    # test_tokenize(x)
    # test_tokenize(y)

    def test_dependency_up():
        x = "The bird flew up in the sky."  # prt (介词后缀)
        x = "The price rises up."  # prt
        x = "He jumped up to touch the bottom of the basketball hoop."  # prt
        x = "He climbed up the mountain."  # prep
        x = "The paper is up to 9 pages."  # prep
        x = "Up until 1973, the ..."  # prep
        x = "There are no trails up the mountain itself."  # prep
        x = "He grew up in a small town."  # prt
        x = "Never give it up."  # prt
        x = "There are no trails up the mountain itself."
        x = "The price is up."  # advmod
        x = "It's up to you."  # prep*
        x = "The sun is up."  # advmod
        x = "He looked up the word in the dictionary."  # prt
        x = "Teachers are up against some major problems these days"  # advmod
        x = "Eve follows them up to the attic room."  # prt
        x = "Further up, resting on the clouds and in a semi-circular formation, ..."  # advmod
        x = "She sang in the Swedish popgroup Granada from the mid-1990s up until their split in 2003."  # advmod
        x = "He looked up the word in the dictionary."  # prt
        x = "He went up to the top of the mountain."  # prt
        x = "There are no trails up the mountain itself."  # prep
        doc = nlp(x)
        for token in doc:
            if token.text == "up":
                # the parent of the up
                parent = token.head
                child = [child for child in token.children]
                print(f"dep of 'up': {token.dep_}\nparent: {parent.text}, parent_lemma: {parent.lemma_}, parent_dep: {parent.ent_type}\nchildren: {[obj.text for obj in child]}\nparent_child: {[(child.text, child.dep_) for child in parent.children]}")
    def test_dependency_down():
        x = "As his body fell down the slope,"  # prep
        x = "DragonVale was acquired by DECA Games in March 2020, after the shut down of Backflip Studios."  # prep
        doc = nlp(x)
        for token in doc:
            if token.text == "down":
                # the parent of the up
                parent = token.head
                child = [child for child in token.children]
                print(f"dep of 'up': {token.dep_}\nparent: {parent.text}, parent_lemma: {parent.lemma_}, parent_dep: {parent.ent_type}\nchildren: {[obj.text for obj in child]}\nparent_child: {[(child.text, child.dep_) for child in parent.children]}")
    # test_dependency_up()
    # test_dependency_down()

    # Process wiki dataset  ========================================================
    # Stage4: annotate sentences with emotions
    # keyword = "down"
    # input_path = DATA_DIR / f"wiki_en/keyword_sentences/wiki_en_{keyword}.txt"
    # output_dir = DATA_DIR / f"wiki_en/keyword_sentences_annotated/"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # model_name = "Qwen/Qwen3-4B-Instruct-2507"  # "Qwen/Qwen3-8B"
    # device = 'cuda:6'
    # batchsize = 256
    # process_data_wiki_stage4(input_path=input_path, output_dir=output_dir, model_name=model_name, device=device, batchsize=batchsize)

    # keyword = "happy"
    # input_path = DATA_DIR / f"wiki_en/keyword_sentences/wiki_en_{keyword}.txt"
    # make_zhipu_infer_file(input_path)

    # Process wiki dataset  ========================================================
    # Stage5: select valid sentences from the output of Stage4
    keyword = "down"
    input_path = DATA_DIR / f"wiki_en/keyword_sentences_annotated/wiki_en_{keyword}.jsonl"
    output_dir = ROOT_DIR / "data/wiki/"
    process_data_wiki_stage5(input_path=input_path, output_dir=output_dir)