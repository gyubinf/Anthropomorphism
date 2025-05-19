

MODEL_CONTEXT_LIMITS = {
    "phi": 4096,
    "qwen": 4096,
    "llama": 8192,
    "deepseek-qwen": 4096,
    "deepseek-llama": 8192
}

def get_context_limit(model_id: str) -> int:
    """
    모델 ID를 기준으로 최대 context 길이를 반환합니다.
    """
    model_id = model_id.lower()
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if key in model_id:
            return limit
    return 4096  # 기본값

def truncate_prompt(prompt: str, max_tokens: int) -> str:
    """
    최대 토큰 수에 맞게 prompt를 자릅니다 (대략적인 문자 수 기준).
    """
    approx_char_limit = max_tokens * 4  # 1 token ≈ 4 characters
    return prompt[:approx_char_limit]
