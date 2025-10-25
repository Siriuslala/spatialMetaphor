import jsonlines


import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))
sys.path.append(root_dir.as_posix())

from data.const import *
def make_metaphor_data_with_context_templates(output_dir, language="cn"):
    lang = language.upper()
    concept_keywords = {
        "emotion_positive": eval(f"EMOTION_POSITIVE_KEWORDS_{lang}"),
        "emotion_negative": eval(f"EMOTION_NEGATIVE_KEWORDS_{lang}"),
        "orientation_positive": eval(f"ORIENTATION_POSTTIVE_KEYWORDS_{lang}"),
        "orientation_negative": eval(f"ORIENTATION_NEGATIVE_KEYWORDS_{lang}")
    }
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
    language = "en"  # "cn" 或 "en"
    output_dir = root_dir / f"data/data_single_context/{language}"
    output_dir.mkdir(parents=True, exist_ok=True)
    make_metaphor_data_with_context_templates(output_dir, language)
