import os
import json
import pandas as pd 
from utils.model_loder import load_model
from utils.utils_prompts import build_prompt
from utils.token_utils import get_context_limit, truncate_prompt
from utils.parsers import parse_sample
from utils.dataset_loader import load_dataset


def save_response(_, sample, prompt, response, model_name: str = None, dataset_name: str = None):
    filename = f"{dataset_name}.jsonl"
    path = os.path.join(
        "./generated_answers/binary_answers",
        model_name,
        filename
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)

    row = {
        "dataset": dataset_name,
        "model_name": model_name,
        "question": sample["question"],
        "choices": sample["choices"],
        "prompt": prompt,
        "response": response
    }

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing_questions = {json.loads(line)["question"] for line in f if line.strip()}
            if row["question"] in existing_questions:
                return

    with open(path, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_response(model, tokenizer, prompt, max_new_tokens=512, backend="transformers", model_id: str = "") -> str:
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


def run_binary(dataset_name: str, model_id: str, output_dir: str,
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

            if parsed_sample["type"] != "binary":
                continue

            full_prompt = build_prompt(parsed_sample, model_id=model_id, backend=backend)
            max_prompt_tokens = context_limit - max_response_tokens
            prompt = truncate_prompt(full_prompt, max_prompt_tokens)

            response = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=max_response_tokens,
                backend=backend,
                model_id=model_id
            )

            save_response(None, parsed_sample, prompt, response,
                          model_name=model_id, dataset_name=dataset_name)


def run_all_models():
    configs = [
        # GPT4All
        ("piqa", "Meta-Llama-3-8B-Instruct.Q4_0.gguf", "./outputs/piqa_llama", "gpt4all"),
        ("winogrande", "Meta-Llama-3-8B-Instruct.Q4_0.gguf", "./outputs/winogrande_phi", "gpt4all"),
        ("boolq", "Meta-Llama-3-8B-Instruct.Q4_0.gguf", "./outputs/boolq_llama", "gpt4all"),

        # DeepSeek-LLaMA
        ("piqa", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "./outputs/piqa_llama8b", "transformers"),
        ("winogrande", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "./outputs/winogrande_llama8b", "transformers"),
        ("boolq", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "./outputs/boolq_llama8b", "transformers"),

        # DeepSeek-Qwen
        ("piqa", "deepseek-ai/DeepSeek-R1-Distill-Qwen-2.0B", "./outputs/piqa_qwen2", "transformers"),
        ("winogrande", "deepseek-ai/DeepSeek-R1-Distill-Qwen-2.0B", "./outputs/winogrande_qwen2", "transformers"),
        ("boolq", "deepseek-ai/DeepSeek-R1-Distill-Qwen-2.0B", "./outputs/boolq_qwen2", "transformers"),

        # Nous-Hermes
        ("piqa", "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf", "./outputs/piqa_phi", "gpt4all"),
        ("winogrande", "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf", "./outputs/winogrande_phi", "gpt4all"),
        ("boolq", "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf", "./outputs/boolq_phi", "gpt4all")
    ]

    for dataset, model_id, outpath, backend in configs:
        run_binary(dataset, model_id, outpath, num_samples=5, backend=backend)


if __name__ == "__main__":
    run_all_models()

