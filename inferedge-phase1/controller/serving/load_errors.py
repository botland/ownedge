"""vLLM load error formatting and failure detection."""

def format_vllm_load_error(record: dict) -> str:
    """Build an actionable operator message from a vLLM exit record."""
    snippet = (record.get("log_snippet") or "").strip()
    lines = [
        line.strip()
        for line in snippet.splitlines()
        if line.strip() and not line.strip().startswith("\x1b")
    ]
    priority = (
        "no available memory for the cache blocks",
        "larger than the maximum number of tokens that can be stored in kv cache",
        "error 804",
        "out of memory",
        "cuda out of memory",
        "oom",
        "valueerror",
        "cuda",
        "nvidia-container-cli",
        "runtimeerror",
        "error",
    )
    for token in priority:
        for line in reversed(lines):
            if token not in line.lower():
                continue
            msg = line[-400:]
            if "cache blocks" in line.lower() or "kv cache" in line.lower():
                msg = (
                    f"{msg} — reduce context_length (e.g. 2048) or raise "
                    "gpu_utilization for this GPU."
                )
            return f"vLLM failed to load model: {msg}"
    exit_code = record.get("exit_code")
    if exit_code is not None:
        return f"vLLM container exited during load (code={exit_code})"
    return "vLLM failed to load model"


def has_vllm_load_failure(config_hash: str, deployment: dict) -> bool:
    """True when this desired config already failed to load and should stay degraded."""
    return (
        deployment.get("config_hash") == config_hash
        and deployment.get("exit_code") not in (None, 0)
    )
