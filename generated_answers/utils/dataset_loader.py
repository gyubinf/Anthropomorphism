import pandas as pd
import os
from typing import List, Dict, Union
from utils.parsers import parse_sample

# 지원하는 확장자 목록
EXTENSIONS = [".jsonl", ".json", ".parquet", ".xml"]

def load_dataset(dataset_name: str, splits: List[str] = ["train", "dev", "test", "val"]) -> pd.DataFrame:
    """
    dataset_name과 여러 splits을 받아 각 split 파일을 불러와 하나의 DataFrame으로 합칩니다.
    """
    base_dir = f"/mnt/netdrive1/llm_datasets_work/dataset{dataset_name}"
    all_data = []

    for split in splits:
        file_loaded = False
        for ext in EXTENSIONS:
            file_path = os.path.join(base_dir, f"{split}{ext}")
            if os.path.exists(file_path):
                if ext == ".jsonl":
                    df = pd.read_json(file_path, lines=True)
                elif ext == ".json":
                    df = pd.read_json(file_path)
                elif ext == ".parquet":
                    df = pd.read_parquet(file_path)
                elif ext == ".xml":
                    df = pd.read_xml(file_path)
                else:
                    continue
                df["__split__"] = split  # split 정보 기록
                all_data.append(df)
                file_loaded = True
                break  # 한 split에 대해 하나의 확장자만 처리
        if not file_loaded:
            print(f"Split '{split}' 파일을 찾을 수 없습니다 in {base_dir}.")

    if not all_data:
        raise FileNotFoundError(f"'{dataset_name}' 데이터셋에 대해 불러온 split이 없습니다.")
    
    return pd.concat(all_data, ignore_index=True)


def parse_dataset(dataset_name: str, splits: List[str] = ["train", "dev", "test", "val"]) -> List[Dict[str, Union[str, List[str]]]]:
    """
    load_dataset으로 불러온 데이터를 parse_sample 함수를 통해 정제
    """
    dataset = load_dataset(dataset_name, splits)
    parsed_samples = []

    for _, sample in dataset.iterrows():
        dataset_type = dataset_name
        parsed_sample = parse_sample(sample.to_dict(), dataset_type)
        parsed_samples.append(parsed_sample)

    return parsed_samples
    