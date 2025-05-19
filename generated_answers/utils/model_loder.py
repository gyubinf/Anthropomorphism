import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpt4all import GPT4All

def load_model(model_id: str, backend: str = "transformers", model_dir: str = "./models"):
    """
    모델과 토크나이저를 불러오는 함수.
    - backend: "transformers" 또는 "gpt4all"
    - model_id: 모델 이름 또는 GPT4All 모델 파일명(.gguf)
    - model_dir: 모델이 저장된 디렉토리
    """
    if backend == "transformers":
        print(f">>> [Transformers] {model_id} 로드 또는 다운로드 중...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model.eval()
            return tokenizer, model
        except Exception as e:
            raise RuntimeError(f"[Transformers] 모델 로드 중 오류 발생: {e}")

    elif backend == "gpt4all":
        # GPT4All 모델은 .gguf 확장자를 가짐
        model_path = os.path.join(model_dir, model_id)
        if not model_id.endswith(".gguf"):
            raise ValueError(f"GPT4All 모델 파일명은 .gguf 확장자를 포함해야 합니다: {model_id}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[GPT4All] 모델 파일이 존재하지 않습니다: {model_path}\n"
                f" 수동 다운로드: https://gpt4all.io/index.html"
            )
        print(f">>> [GPT4All] {model_id} 로드 중...")
        model = GPT4All(model_name=model_id, model_path=model_dir)
        return None, model  # GPT4All은 tokenizer 불필요

    else:
        raise ValueError(f"지원하지 않는 backend: {backend}")

