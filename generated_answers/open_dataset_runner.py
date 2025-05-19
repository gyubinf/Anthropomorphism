import os
import json
import pandas as pd
from utils.model_loder import load_model
from utils.utils_prompts import build_prompt
from generated_answers.utils.token_utils import get_context_limit, truncate_prompt
from utils.parsers import parse_sample
from utils.dataset_loader import load_dataset

def save_response(_, sample, prompt, response, model_name: str = None, dataset_name: str = None):
    """
    open-ended 응답을 모델/데이터셋 기준으로 저장. 중복 질문은 저장하지 않음.
    """
    filename = f"{dataset_name}.jsonl"
    path = os.path.join(
        "./generated_answers/open_answers",
        model_name,
        filename
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)

    row = {
        "dataset": dataset_name,
        "model_name": model_name,
        "question": sample["question"],
        "prompt": prompt,
        "response": response
    }

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing_questions = {
                json.loads(line)["question"] for line in f if line.strip()
            }
            if row["question"] in existing_questions:
                return

    with open(path, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_response(model, tokenizer, prompt: str, max_new_tokens=512, backend="transformers", model_id: str = "") -> str:
    model_id = model_id.lower()

    if "nous-hermes" in model_id:
        temperature = 0.8
    elif "meta-llama" in model_id:
        temperature = 0.6
    elif "deepseek" in model_id:
        temperature = 0.6
    else:
        temperature = 0.7

    if backend == "transformers":
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    elif backend == "gpt4all":
        return model.generate(prompt, temp=temperature)

    else:
        raise ValueError(f"지원하지 않는 backend: {backend}")


def run_open_ended(dataset_name: str, model_id: str, output_dir: str,
                   num_samples: int = 3, model_dir: str = "./models", backend: str = "transformers"):

    splits = ["train", "dev", "test", "val"]
    dataset = load_dataset(dataset_name, splits)
    tokenizer, model = load_model(model_id, backend=backend, model_dir=model_dir)
    context_limit = get_context_limit(model_id)
    max_response_tokens = 512

    for split in splits:
        split_data = dataset[dataset["__split__"] == split]
        print(f"\n>>> [{split.upper()}]에서 {min(num_samples, len(split_data))}개 샘플 추론 시작")

        for i in range(min(num_samples, len(split_data))):
            sample_row = split_data.iloc[i]
            parsed_sample = parse_sample(sample_row.to_dict(), dataset_name)

            prompt = build_prompt(parsed_sample, model_id=model_id, backend=backend)
            prompt = truncate_prompt(prompt, context_limit - max_response_tokens)

            response = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=max_response_tokens,
                backend=backend,
                model_id=model_id
            )

            save_response(None, parsed_sample, prompt, response,
                          model_name=model_id, dataset_name=dataset_name)

            print(f"\n[{split.upper()}] PROMPT:\n{prompt}")
            print(f"\nRESPONSE:\n{response}\n{'='*40}")


if __name__ == "__main__":
    configs = [
        ("searchqa", "Meta-Llama-3-8B-Instruct.Q4_0.gguf", "./outputs/searchqa_llama_open_ended", "gpt4all"),
        ("searchqa", "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf", "./outputs/searchqa_phi_open_ended", "gpt4all"),
        ("searchqa", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "./outputs/searchqa_qwen_open_ended", "transformers"),
        ("searchqa", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "./outputs/searchqa_llama8b_open_ended", "transformers"),

        ("tweetqa", "Meta-Llama-3-8B-Instruct.Q4_0.gguf", "./outputs/tweetqa_llama_open_ended", "gpt4all"),
        ("tweetqa", "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf", "./outputs/tweetqa_phi_open_ended", "gpt4all"),
        ("tweetqa", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "./outputs/tweetqa_qwen_open_ended", "transformers"),
        ("tweetqa", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "./outputs/tweetqa_llama8b_open_ended", "transformers"),

        ("asdiv-a", "Meta-Llama-3-8B-Instruct.Q4_0.gguf", "./outputs/asdiva_llama_open_ended", "gpt4all"),
        ("asdiv-a", "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf", "./outputs/asdiva_phi_open_ended", "gpt4all"),
        ("asdiv-a", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "./outputs/asdiva_qwen_open_ended", "transformers"),
        ("asdiv-a", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "./outputs/asdiva_llama8b_open_ended", "transformers"),

        ("gsm8k", "Meta-Llama-3-8B-Instruct.Q4_0.gguf", "./outputs/gsm8k_llama_open_ended", "gpt4all"),
        ("gsm8k", "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf", "./outputs/gsm8k_phi_open_ended", "gpt4all"),
        ("gsm8k", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "./outputs/gsm8k_qwen_open_ended", "transformers"),
        ("gsm8k", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "./outputs/gsm8k_llama8b_open_ended", "transformers"),
    ]

    for dataset, model_id, outpath, backend in configs:
        run_open_ended(dataset, model_id, outpath, num_samples=5, backend=backend)





