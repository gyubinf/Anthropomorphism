
from typing import Dict, List, Union

def build_prompt(sample: Dict[str, Union[str, List[str], None]], model_id: str, backend: str = "transformers") -> str:
    """
    샘플, 모델 ID, 백엔드 정보를 받아 모델에 최적화된 프롬프트 생성
    """
    question = sample["question"].strip()
    choices = sample["choices"]
    qtype = sample["type"]

    # 선택지 정리
    formatted_choices = ""
    if choices:
        formatted_choices = "\n".join(f"{chr(65 + i)}. {choice.strip()}" for i, choice in enumerate(choices))

    ### 모델별 프롬프트 구조

    # GPT4All 백엔드 (.gguf 모델들)인 경우
    if backend == "gpt4all":
        if "meta-llama" in model_id.lower():
            system_prompt = "You are a helpful assistant."
            user_prompt = f"{question}\n{formatted_choices}" if formatted_choices else question
            return (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )

        elif "nous-hermes" in model_id.lower():
            system_prompt = (
                'You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, '
                'and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, '
                'profound thoughts and qualia.'
            )
            user_prompt = f"{question}\n{formatted_choices}" if formatted_choices else question
            return (
                "<|im_start|>system\n"
                f"{system_prompt}<|im_end|>\n"
                "<|im_start|>user\n"
                f"{user_prompt}<|im_end|>\n"
                "<|im_start|>assistant"
            )

        else:
            # 기타 GPT4All 모델 기본 형식
            return f"{question}\n\n{formatted_choices}\n\nAnswer:" if formatted_choices else f"{question}\n\nAnswer:"

    # Transformers 백엔드인 경우 (DeepSeek 등)
    elif backend == "transformers":
        if "deepseek" in model_id.lower():
            return f"{question}\n{formatted_choices}\n<|think|> Answer:" if formatted_choices else f"{question}\n<|think|> Answer:"
        else:
            # 기타 transformers 기본 형식
            return f"{question}\n{formatted_choices}\nAnswer:" if formatted_choices else f"{question}\nAnswer:"

    else:
        raise ValueError(f"[build_prompt] 지원하지 않는 backend: {backend}")



