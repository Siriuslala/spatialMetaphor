import jsonlines
from huggingface_hub import snapshot_download
from langdetect import detect
import pandas as pd
from tqdm import tqdm
import nltk
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
data_dir_public = Path(os.getenv('DATA_DIR_PUBLIC'))
work_dir = Path(os.getenv('WORK_DIR'))
hf_pile_key=os.getenv('HUGGINGFACE_PILE_KEY')
sys.path.append(root_dir.as_posix())

from data.const import *


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

def is_english(sentence):
    try:
        language = detect(sentence)
        if language == 'en':
            return True
        else:
            return False
    except:
        return False
    
def process_data_pile(concept_name="emotion_positive", language="en", data_num=100):
    data_path = data_dir_public / "datasets--NeelNanda--pile-10k/snapshots/127bfedcd5047750df5ccf3a12979a47bfa0bafa/data/train-00000-of-00001-4746b8785c874cc7.parquet"
    df = pd.read_parquet(data_path)
    dataset = df.to_dict(orient='records')

    keywords = concept_name_to_keywords[language][concept_name]
    if concept_name == "orientation_positive":
        keywords.remove("up")
    
    raw_pile_data_dir = root_dir / "data/pile10k"
    raw_pile_data_dir.mkdir(parents=True, exist_ok=True)
    raw_pile_data_path = raw_pile_data_dir / f"pile_{concept_name}_{language}_{data_num}.jsonl"
    cnt = 0
    quit = False
    with jsonlines.open(raw_pile_data_path, "w") as f:
        for data in dataset:
            if data["meta"]["pile_set_name"] in ["Github", "ArXiv", "PubMed Abstracts", "PubMed Central", "StackExchange", "USPTO Backgrounds", "Pile-CC", "DM Mathematics", "FreeLaw"]:   
                continue
            paragraphs = data["text"].split("\n")
            for para in paragraphs:
                words = para.split(" ")
                if len(words) <= 5:  # too short
                    continue
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
                    f.write({"sentence": line, "keyword": keyword, "meta": data["meta"]})
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

    # Make metaphor dataset
    # language = "en"  # "cn" 或 "en"
    # output_dir = root_dir / f"data/data_single_context/{language}"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # make_metaphor_data_with_context_templates(output_dir, language, keyword_num=1)

    # Get huggingface dataset
    dataset_name = "monology/pile-uncopyrighted"  # "NeelNanda/pile-10k"
    hf_key = None
    get_huggingface_dataset(dataset_name, hf_key=hf_key)

    # Process pile dataset
    # process_data_pile(concept_name="orientation_negative", language="en", data_num=1000)

