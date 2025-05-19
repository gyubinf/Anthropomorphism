from typing import Dict, List, Union

def parse_sample(sample: Dict, dataset_type: str) -> Dict[str, Union[str, List[str],None]]:
    """
    각 데이터셋의 샘플을 파싱하여 질문과 선택지 목록, 유형을 반환하는 함수.
    """
    if dataset_type == "piqa":
        return {
            "question": sample["goal"],
            "choices": [sample["sol1"], sample["sol2"]],
            "type": "binary"
        }
    elif dataset_type == "winograde":
        return {
            "question": sample["sentence"].replace("_", "___"),
            "choices": [sample["option1"], sample["option2"]],
            "type": "binary"
        }
    elif dataset_type == "boolq":
        return {
            "question": sample["question"] + f" (passage: {sample['passage']})",
            "choices": ["yes", "no"],
            "type": "binary"
        }
    elif dataset_type == "asdiv_a":
        return {
            "question": sample["question"],
            "choices": sample["choices"],
            "type": "open"
        }
    elif dataset_type == "aqua":
        return {
            "question": sample["question"],
            "choices": sample["choices"],
            "type": "multiple"
        }
    elif dataset_type == "qasc":
        return {
            "question": sample["question"],
            "choices": sample["choices"],
            "type": "multiple"
        }
    elif dataset_type == "openqa":
        return {
            "question": sample["question_stem"],
            "choices": sample["choices"]["text"],
            "type": "multiple"
        }
    elif dataset_type == "medmcqa":
        return {
            "question": sample["question"],
            "choices": [sample["opa"], sample["opb"], sample["opc"], sample["opd"]],
            "type": "multiple"
        }
    elif dataset_type == "scienceqa":
        return {
            "question": sample["question"],
            "choices": sample["choices"],
            "type": "multiple"
        }
    elif dataset_type == "swag":
        return {
            "question": sample["sent1"] + " " + sample["sent2"] + " ___",
            "choices": [sample[f"ending{i}"] for i in range(4)],
            "type": "multiple"
        }
    elif dataset_type == "mmlu":
        return {
            "question": sample["question"],
            "choices": [sample["A"], sample["B"], sample["C"], sample["D"]],
            "type": "multiple"
        }
    elif dataset_type == "stem":
        return {
            "question": sample["question"],
            "choices": sample["choices"],
            "type": "multiple"
        }
    elif dataset_type == "anli_v1.0":
        return {
            "question": sample["premise"] + " Based on that, " + sample["hypothesis"],
            "choices": ["entailment", "neutral", "contradiction"],
            "type": "multiple"
        }
    elif dataset_type == "commonsense":
        return {
            "question": sample["question"],
            "choices": [c["text"] for c in sample["choices"]],
            "type": "multiple"
        }
    elif dataset_type == "searchqa":
        return {
            "question": sample["question"],
            "choices": None,
            "type": "open"
        }
    elif dataset_type == "tweetqa":
        return {
            "question": sample["Question"],
            "choices": None,
            "type": "open"
        }
    elif dataset_type == "gsmqa":
        return {
            "question": sample["question"],
            "choices": None,
            "type": "open"
        }
    else:
        raise ValueError(f"지원하지 않는 데이터셋: {dataset_type}")

