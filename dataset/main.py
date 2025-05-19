import os
import requests # HTTP를 연결해주는 코드 
import zipfile
import json
import subprocess # 외부 프로그램을 실행하거나, 명령어를 실행하여 결과를 다룰 수 있는 코드 
import shutil # 파일 시스템 작업을 간편하게 처리 할수 있게 도와주는 코드 

def download_file(url, save_path):
    """서버에서 파일을 URL로 다운로드하고 지정된 경로에 저장하는 함수"""
    print(f"Downloading {url}...")
    response = requests.get(url)
    if response.status_code == 200: # HTTP 응답 상태 코드를 의미, 200: HTTP이미 정해진 코드이며, 응답 ok를 뜻을 의미 
        with open(save_path, 'wb') as f: # 파일 경로를 보내고 파일을 바이너리 형태로 저장을 한다라는 뜻 
            f.write(response.content)
        print(f"Saved to {save_path}")
    else:
        print(f"Failed to download {url}")

def extract_zip_file(zip_path, extract_to):
    """ZIP 파일을 지정된 디렉터리에 압축 해제"""
    if not os.path.exists(zip_path):
        print(f"Error: ZIP file {zip_path} not found!")
        return False  # 오류 반환
    
    if os.path.exists(extract_to):
        print(f"Skipping extraction: {extract_to} already exists.")
        return True  # 이미 압축이 풀려 있으면 건너뛰기
    
    try:
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction completed successfully!")
        return True
    except zipfile.BadZipFile:
        print(f"Error: The ZIP file {zip_path} is corrupted.")
        return False
    except PermissionError:
        print(f"Error: Permission denied for extracting to {extract_to}. Try running with sudo.")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def load_jsonl(file_path):
    """jsonl 파일을 서버에서 읽어오는 함수"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def download_git_lfs(url, save_path):
    """Git LFS를 통해 파일을 다운로드"""
    print(f"Downloading from Git LFS: {url}...")
    subprocess.run(["git", "lfs", "clone", url, save_path], check=True)


def download_winograde(server_dir):
    """Winograde 데이터셋 다운로드 및 추출"""
    winograde_url = "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip"  # 수정된 URL
    zip_path = os.path.join(server_dir, "winograde.zip")
    extract_folder = os.path.join(server_dir, "winograde")

    # 파일 다운로드
    response = requests.get(winograde_url)
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded winograde.zip to {zip_path}")
        
        # ZIP 파일 추출
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)    
        print(f"Extracted winograde.zip to {extract_folder}")
    else:
        print(f"Failed to download {winograde_url}, status code: {response.status_code}")


def download_anli(server_dir):
    """ANLI 데이터셋 다운로드 및 추출"""
    anli_url = "https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip"  # 실제 URL로 수정 필요
    zip_path = os.path.join(server_dir, "anli.zip")
    extract_folder = os.path.join(server_dir, "anli")

    # 다운로드 및 추출
    if not os.path.exists(zip_path):
        download_file(anli_url, zip_path)

    if not os.path.exists(extract_folder):
        extract_zip_file(zip_path, extract_folder)
        

    # jsonl 파일 로드
    train_data = load_jsonl(os.path.join(extract_folder, "train.jsonl"),encoding="utf-8")
    dev_data = load_jsonl(os.path.join(extract_folder, "dev.jsonl"),encoding="utf-8")
    test_data = load_jsonl(os.path.join(extract_folder, "test.jsonl"),encoding="utf-8")
    
    return train_data, dev_data, test_data



def download_piqa(url, save_path):
    """PIQA 데이터셋 다운로드"""
    print(f"Downloading {url}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Saved to {save_path}")
    else:
        print(f"Failed to download {url}, status code: {response.status_code}")

# PIQA 데이터셋의 raw URL
piqa_train_url = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/refs/heads/master/piqa/data/train.jsonl"
piqa_test_url = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/refs/heads/master/piqa/data/tests.jsonl"
piqa_val_url = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/refs/heads/master/piqa/data/valid.jsonl"

# 저장할 경로
server_dir = "/media/hail/HDD/llm_datasets/dataset"
download_piqa(piqa_train_url, f"{server_dir}/piqa_train.jsonl")
download_piqa(piqa_test_url, f"{server_dir}/piqa_test.jsonl")
download_piqa(piqa_val_url, f"{server_dir}/piqa_dev.jsonl")


def download_git_lfs_data(server_dir, dataset_name, git_lfs_url):
    """Git LFS에서 데이터셋을 다운로드하는 함수 (PARQUET 형식 또는 ZIP 형식)"""
    dataset_dir = os.path.join(server_dir, dataset_name)
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    download_git_lfs(git_lfs_url, dataset_dir)
    
    return dataset_dir


def download_asdiva(server_dir):
    """ASDiv-A 데이터셋을 GitHub에서 다운로드하고 XML로 저장"""
    asdiva_url = "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/refs/heads/master/dataset/ASDiv.xml" # 실제 URL로 수정 필요
    xml_path = os.path.join(server_dir, "asdiv-a.xml")

    # 다운로드
    if not os.path.exists(xml_path):
        download_file(asdiva_url, xml_path)

    # XML 파일 로드 (필요한 처리를 추가)
    return xml_path

def download_parquet_data(server_dir, dataset_name, file_url):
    """PARQUET 파일을 다운로드하여 서버에 저장"""
    dataset_dir = os.path.join(server_dir, dataset_name)
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    file_name = file_url.split("/")[-1]
    save_path = os.path.join(dataset_dir, file_name)
    
    # 파일 다운로드
    download_file(file_url, save_path)
    
    return save_path

def download_boolq(server_dir):
    """BoolQ 데이터셋 다운로드 (train, validation)"""
    boolq_train_url = "https://huggingface.co/datasets/google/boolq/resolve/main/data/train-00000-of-00001.parquet"
    boolq_val_url = "https://huggingface.co/datasets/google/boolq/resolve/main/data/validation-00000-of-00001.parquet"

    train_path = download_parquet_data(server_dir, "boolq", boolq_train_url)
    val_path = download_parquet_data(server_dir, "boolq", boolq_val_url)

    return train_path, val_path

def download_commonsense(server_dir):
     """ Commonsense 데이터셋 다운로드(train,validation, test)"""
     commonsense_test_url="https://huggingface.co/datasets/tau/commonsense_qa/resolve/main/data/test-00000-of-00001.parquet"
     commonsense_train_url="https://huggingface.co/datasets/tau/commonsense_qa/resolve/main/data/train-00000-of-00001.parquet"
     commonsense_val_url="https://huggingface.co/datasets/tau/commonsense_qa/resolve/main/data/validation-00000-of-00001.parquet"
     
     train_path=download_parquet_data(server_dir,"commonsense",commonsense_train_url)
     val_path= download_parquet_data(server_dir,"commonsense",commonsense_val_url)
     test_path=download_parquet_data(server_dir,"commonsense",commonsense_test_url)
     
     return train_path,val_path,test_path
     
def download_qasc(server_dir):
    """ qasc 데이터셋 다운로드(train,validation,test)"""
    qasc_train_url="https://huggingface.co/datasets/allenai/qasc/resolve/main/data/train-00000-of-00001.parquet"
    qasc_val_url="https://huggingface.co/datasets/allenai/qasc/resolve/main/data/validation-00000-of-00001.parquet"
    qasc_test_url="https://huggingface.co/datasets/allenai/qasc/resolve/main/data/test-00000-of-00001.parquet"
    
    train_path=download_parquet_data(server_dir,"qasc",qasc_train_url)
    val_path=download_parquet_data(server_dir,"qasc",qasc_val_url)
    test_path=download_parquet_data(server_dir,"qasc",qasc_test_url)
    
    return train_path,val_path,test_path

def download_openqa(server_dir):
    """ openqa 데이터셋 다운로드(train,validation,test)"""
    openqa_train_url="https://huggingface.co/datasets/allenai/openbookqa/resolve/main/main/train-00000-of-00001.parquet"
    openqa_val_url="https://huggingface.co/datasets/allenai/openbookqa/resolve/main/main/validation-00000-of-00001.parquet"
    openqa_test_url="https://huggingface.co/datasets/allenai/openbookqa/resolve/main/main/test-00000-of-00001.parquet"
    
    train_path=download_parquet_data(server_dir,"openqa",openqa_train_url)
    val_path=download_parquet_data(server_dir,"openqa",openqa_val_url)
    test_path=download_parquet_data(server_dir,"openqa",openqa_test_url)
    
    return train_path,val_path,test_path

def download_medmcqa(server_dir):
    """ medmcqa 데이터셋 다운로드(train,validation,test)"""
    medmcqa_train_url="https://huggingface.co/datasets/openlifescienceai/medmcqa/resolve/main/data/train-00000-of-00001.parquet"
    medmcqa_val_url="https://huggingface.co/datasets/openlifescienceai/medmcqa/resolve/main/data/validation-00000-of-00001.parquet"
    medmcqa_test_url="https://huggingface.co/datasets/openlifescienceai/medmcqa/resolve/main/data/test-00000-of-00001.parquet"
    
    train_path=download_parquet_data(server_dir,"medmcqa",medmcqa_train_url)
    val_path=download_parquet_data(server_dir,"medmcqa",medmcqa_val_url)
    test_path=download_parquet_data(server_dir,"medmcqa",medmcqa_test_url)
    
    return train_path,val_path,test_path
    
def download_scienceqa(server_dir):
    """ scienceqa 데이터셋 다운로드(train,validation,test)"""
    scienceqa_train_url="https://huggingface.co/datasets/derek-thomas/ScienceQA/resolve/main/data/train-00000-of-00001-1028f23e353fbe3e.parquet"
    scienceqa_val_url="https://huggingface.co/datasets/derek-thomas/ScienceQA/resolve/main/data/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    scienceqa_test_url="https://huggingface.co/datasets/derek-thomas/ScienceQA/resolve/main/data/test-00000-of-00001-f0e719df791966ff.parquet"
    
    train_path=download_parquet_data(server_dir,"scienceqa",scienceqa_train_url)
    val_path=download_parquet_data(server_dir,"scienceqa",scienceqa_val_url)
    test_path=download_parquet_data(server_dir,"scienceqa",scienceqa_test_url)
    
    return train_path,val_path,test_path

def download_swag(server_dir):
    """ swag 데이터셋 다운로드(train,validation,test)"""
    swag_train_url="https://huggingface.co/datasets/allenai/swag/resolve/main/full/train-00000-of-00001.parquet"
    swag_val_url="https://huggingface.co/datasets/allenai/swag/resolve/main/full/validation-00000-of-00001.parquet"
    
    train_path=download_parquet_data(server_dir,"swag",swag_train_url)
    val_path=download_parquet_data(server_dir,"swag",swag_val_url)
    
    return train_path,val_path

def download_mmlu(server_dir):
    """ mmlu 데이터셋 다운로드(train,validation,test)"""
    mmlu_dev_url="https://huggingface.co/datasets/cais/mmlu/resolve/main/all/dev-00000-of-00001.parquet"
    mmlu_val_url="https://huggingface.co/datasets/cais/mmlu/resolve/main/all/validation-00000-of-00001.parquet"
    mmlu_test_url="https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet"
    
    dev_path=download_parquet_data(server_dir,"mmlu",mmlu_dev_url)
    val_path=download_parquet_data(server_dir,"mmlu",mmlu_val_url)
    test_path=download_parquet_data(server_dir,"mmlu",mmlu_test_url)
    
    return dev_path,val_path,test_path
    
def download_stem(server_dir):
    """ stem 데이터셋 다운로드(train,validation,test)"""
    stem_train_url="https://huggingface.co/datasets/stemdataset/STEM/resolve/main/data/train-00004-of-00058.parquet"
    stem_val_url="https://huggingface.co/datasets/stemdataset/STEM/resolve/main/data/valid-00000-of-00020.parquet"
    stem_test_url="https://huggingface.co/datasets/stemdataset/STEM/resolve/main/data/test-00000-of-00020.parquet"
    
    train_path=download_parquet_data(server_dir,"stem",stem_train_url)
    val_path=download_parquet_data(server_dir,"stem",stem_val_url)
    test_path=download_parquet_data(server_dir,"stem",stem_test_url)
    
    return train_path,val_path, test_path

def download_gsmqa(server_dir):
    """ gsmqa데이터셋 다운로드(train,test)"""
    gsmqa_train_url="https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet"
    gsmqa_test_url="https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"
    
    train_path=download_parquet_data(server_dir,"gsmqa",gsmqa_train_url)
    test_path=download_parquet_data(server_dir,"gsmqa",gsmqa_test_url)
    
    return train_path,test_path

# 서버 디렉터리 및 저장소 URL 설정
server_dir = "/media/hail/HDD/llm_datasets/dataset"  
searchqa_repo_url = "https://huggingface.co/datasets/kyunghyuncho/search_qa/resolve/main/data/train_test_val"  
searchqa_dir = os.path.join(server_dir, "searchqa")
os.makedirs(searchqa_dir,exist_ok=True)


def download_and_extract_zip(url, zip_path, extract_to):
    """ZIP 파일 다운로드 및 압축 해제"""
    if not os.path.exists(zip_path):
        print(f"Downloading {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            print(f"Saved to {zip_path}")
        else:
            print(f"Failed to download {url}, status code: {response.status_code}")
            return
    else:
        print(f"{zip_path} already exists. Skipping download.")
    
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction complete.")
    else:
        print(f"Extraction folder {extract_to} already exists. Skipping extraction.")

def process_searchqa_dataset():
    """SearchQA 데이터셋 다운로드 및 압축 해제"""
    zip_files = ["train.zip", "test.zip", "val.zip"]
    
    for zip_file in zip_files:
        url = f"{searchqa_repo_url}/{zip_file}"
        zip_path = os.path.join(searchqa_dir, zip_file)
        extract_folder = os.path.join(searchqa_dir, zip_file.replace(".zip", ""))
        download_and_extract_zip(url, zip_path, extract_folder)

# 실행
process_searchqa_dataset()


def main():
    server_dir = "/media/hail/HDD/llm_datasets/dataset"

    #anli 다운로드 

    
    # PIQA 다운로드
    download_piqa(piqa_train_url, f"{server_dir}/piqa_train.jsonl")
    download_piqa(piqa_test_url, f"{server_dir}/piqa_test.jsonl")
    download_piqa(piqa_val_url, f"{server_dir}/piqa_val.jsonl")
    
    # Winograde 다운로드
    download_winograde(server_dir)
    
    # ASDiv-A 다운로드
    download_asdiva(server_dir)
    
    # SEARCHQA 데이터 다운로드 및 처리

    process_searchqa_dataset()
    
    # 기타 PARQUET 데이터셋 다운로드
    download_boolq(server_dir)
    download_commonsense(server_dir)
    download_qasc(server_dir)
    download_openqa(server_dir)
    download_medmcqa(server_dir)
    download_scienceqa(server_dir)
    download_swag(server_dir)
    download_mmlu(server_dir)
    download_stem(server_dir)
    download_gsmqa(server_dir)
    
    print("All datasets downloaded and processed successfully.")

if __name__ == "__main__":
    main()
    
    




