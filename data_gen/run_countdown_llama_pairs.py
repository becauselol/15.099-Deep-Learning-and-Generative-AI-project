import os
import ast
import math
import random
import argparse
import re
import time
from itertools import combinations  # kept if you want later features
from typing import Optional

import pandas as pd
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


# ============================
# Llama singleton via HF Hub
# ============================

_LLM_SINGLETON = None
_LLM_SINGLETON_KEY = None


def _get_llm_from_hf(
    repo_id: str,
    filename: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    seed: int = 0,
    verbose: bool = False,
    token: Optional[str] = None,
) -> Llama:
    """
    Lazy-init a single Llama instance from a GGUF on Hugging Face Hub.

    - Downloads the GGUF file (or reuses local cache).
    - Instantiates llama_cpp.Llama with the given context / GPU config.
    - Reuses the same Llama instance across calls if key matches.
    """
    global _LLM_SINGLETON, _LLM_SINGLETON_KEY

    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,  # can be None for public repos
    )

    key = (model_path, int(n_ctx), int(n_gpu_layers), int(seed), bool(verbose))
    if _LLM_SINGLETON is None or _LLM_SINGLETON_KEY != key:
        _LLM_SINGLETON = Llama(
            model_path=os.path.abspath(os.path.expanduser(model_path)),
            n_ctx=int(n_ctx),
            n_gpu_layers=int(n_gpu_layers),
            seed=int(seed),
            logits_all=False,
            verbose=bool(verbose),
        )
        _LLM_SINGLETON_KEY = key

    return _LLM_SINGLETON


def call_llama_local(
    prompt: str,
    *,
    repo_id: str,
    filename: str,
    max_tokens: int,
    temperature: float,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    seed: int = 0,
    retries: int = 3,
    backoff_s: int = 2,
    verbose: bool = False,
) -> str:
    """
    Local GPU chat using llama_cpp on a compute node, with the model pulled from HF Hub.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            llm = _get_llm_from_hf(
                repo_id=repo_id,
                filename=filename,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                seed=seed,
                verbose=verbose,
            )
            resp = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            choices = resp.get("choices") or []
            return choices[0]["message"]["content"] if choices else ""
        except Exception as e:
            last_err = e
            if attempt == retries:
                raise
            time.sleep(backoff_s * attempt)
    if last_err:
        raise last_err
    return ""


# -----------------------------
# Helpers: formatting & loading
# -----------------------------

def format_numbers_for_prompt(numbers):
    """
    Format numbers to inject into the prompt.
    Here we use the style: [44 19 35]
    """
    return "[" + " ".join(str(x) for x in numbers) + "]"


def load_prompts(prompts_file):
    df = pd.read_csv(prompts_file)
    # Ensure there is a prompt_id column called "id"
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    if "prompt" not in df.columns:
        raise ValueError(f"{prompts_file} must have a 'prompt' column.")
    return df[["id", "prompt"]]


def load_instances(instances_file):
    df = pd.read_csv(instances_file)
    # Expect at least: id, numbers, target
    if "id" not in df.columns or "numbers" not in df.columns or "target" not in df.columns:
        raise ValueError(f"{instances_file} must have 'id', 'numbers', and 'target' columns.")
    return df[["id", "numbers", "target"]]


def init_or_load_pairs(pairs_file, prompts_df, instances_df, num_pairs):
    """
    If pairs_file exists: load and return.
    Otherwise: create a random sample of prompt/instance pairs and return a new DataFrame.
    Columns: prompt_id, instance_id, response, correct
    """
    if os.path.exists(pairs_file):
        return pd.read_csv(pairs_file)

    # Create all possible pairs (could be large – sample)
    all_pairs = [(int(p_id), int(i_id)) for p_id in prompts_df["id"] for i_id in instances_df["id"]]
    if num_pairs is None or num_pairs <= 0 or num_pairs >= len(all_pairs):
        sampled_pairs = all_pairs
    else:
        sampled_pairs = random.sample(all_pairs, num_pairs)

    df_pairs = pd.DataFrame(sampled_pairs, columns=["prompt_id", "instance_id"])
    df_pairs["response"] = ""
    df_pairs["correct"] = pd.NA
    return df_pairs


# -----------------------------
# Robust answer checker
# -----------------------------

def safe_eval_expression(expr):
    """
    Evaluate a numeric expression consisting only of digits, +, -, *, /, parentheses, and whitespace.
    We assume the expression has already been validated to contain only safe characters.
    """
    expr = expr.strip()
    return eval(expr, {"__builtins__": None}, {})


def check_answer(numbers, target, response):
    """
    More robust checker:
    - Strips <think>...</think> blocks (all occurrences).
    - Extracts the LAST <answer>...</answer>.
    - Allows an optional 'expr = target' form.
    - Ensures only allowed characters in the answer expression.
    - Ensures each given number is used exactly once (multiset equality, based on the LHS if '=' is present).
    - Evaluates expression and checks equality with target.

    Returns (correct: bool, message: str)
    """
    if response is None:
        return False, "Empty response."

    text = str(response)

    # Remove all <think>...</think> blocks (non-greedy)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    start_tag = "<answer>"
    end_tag = "</answer>"

    start_idx = text.rfind(start_tag)
    end_idx = text.rfind(end_tag)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return False, "Answer tags not found or malformed."

    answer_content = text[start_idx + len(start_tag):end_idx].strip()
    if not answer_content:
        return False, "Answer content is empty."

    # Allow digits, whitespace, parentheses, + - * / and optionally '='
    if not re.fullmatch(r"[0-9\s\+\-\*\/\(\)=]+", answer_content):
        return False, "Invalid characters in answer expression."

    # Handle optional "expr = target" pattern
    lhs_expr = answer_content
    rhs_str = None

    if "=" in answer_content:
        parts = answer_content.split("=")
        if len(parts) != 2:
            return False, "More than one '=' found in answer expression."
        lhs_expr = parts[0].strip()
        rhs_str = parts[1].strip()
        if not lhs_expr:
            return False, "Left-hand side of '=' is empty."
        if not rhs_str:
            return False, "Right-hand side of '=' is empty."

        # rhs must be an integer matching the target
        try:
            rhs_val = int(rhs_str)
        except Exception:
            return False, f"Right-hand side '{rhs_str}' is not an integer."

        try:
            target_int = int(target)
        except Exception:
            return False, f"Target '{target}' is not an integer."

        if rhs_val != target_int:
            return False, f"Right-hand side of equation is {rhs_val}, target is {target_int}."

    # Only allow digits/op/parens/whitespace on LHS too
    if not re.fullmatch(r"[0-9\s\+\-\*\/\(\)]+", lhs_expr):
        return False, "Invalid characters in left-hand side expression."

    # Parse numbers from the original instance and from the LHS expression
    inst_nums = list(map(int, numbers))
    used_nums = list(map(int, re.findall(r"\d+", lhs_expr)))

    if not used_nums:
        return False, "No numbers found in answer expression."

    from collections import Counter
    inst_counts = Counter(inst_nums)
    used_counts = Counter(used_nums)

    # Check for extra or overused numbers
    for num, cnt in used_counts.items():
        if num not in inst_counts:
            return False, f"Used number {num} not in provided numbers."
        if cnt > inst_counts[num]:
            return False, f"Number {num} used {cnt} times but only {inst_counts[num]} available."

    # Check for missing numbers
    for num, cnt in inst_counts.items():
        if used_counts.get(num, 0) != cnt:
            return False, f"Number {num} used {used_counts.get(num,0)} times but should be {cnt}."

    # Evaluate LHS expression
    try:
        value = safe_eval_expression(lhs_expr)
    except Exception as e:
        return False, f"Error evaluating expression: {e}"

    try:
        target_int = int(target)
    except Exception:
        return False, f"Target '{target}' is not an integer."

    # Numeric equality with small tolerance
    if isinstance(value, float):
        correct = abs(value - target_int) < 1e-6
    else:
        correct = (value == target_int)

    if not correct:
        return False, f"Expression evaluates to {value}, target is {target_int}."
    return True, ""

# -----------------------------
# Main loop
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Run local Llama on random (prompt, instance) Countdown pairs.")
    parser.add_argument("--prompts_file", type=str, default="transformed_countdown_prompts_gemini.csv")
    parser.add_argument("--instances_file", type=str, default="countdown_features.csv")
    parser.add_argument("--pairs_file", type=str, default="countdown_prompt_instance_results.csv")
    parser.add_argument("--num_pairs", type=int, default=1000,
                        help="Number of random prompt-instance pairs to generate if pairs_file does not exist.")
    parser.add_argument("--max_calls", type=int, default=50,
                        help="Maximum number of new LLM calls to make in this run.")

    # Model params (Hugging Face GGUF)
    parser.add_argument("--repo_id", type=str, default="TheBloke/Llama-2-7B-Chat-GGUF",
                        help="Hugging Face repo id containing the GGUF file.")
    parser.add_argument("--filename", type=str, default="llama-2-7b-chat.Q4_K_M.gguf",
                        help="GGUF filename within the repo.")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--n_ctx", type=int, default=4096)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--seed_local", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Seed randomness for reproducibility of pair sampling
    random.seed(42)

    # 1. Load prompts and instances
    prompts_df = load_prompts(args.prompts_file)
    instances_df = load_instances(args.instances_file)

    # 2. Init or load prompt-instance pairs
    pairs_df = init_or_load_pairs(args.pairs_file, prompts_df, instances_df, args.num_pairs)

    # Ensure "message" column exists
    if "message" not in pairs_df.columns:
        pairs_df["message"] = ""

    # 3. Determine which rows still need responses
    pending_mask = pairs_df["response"].isna() | (pairs_df["response"].astype(str).str.len() == 0)
    pending_indices = pairs_df.index[pending_mask].tolist()
    random.shuffle(pending_indices)

    if not pending_indices:
        print("No pending (prompt, instance) pairs left to evaluate.")
        return

    to_process = pending_indices[: args.max_calls]
    print(f"Processing {len(to_process)} pairs this run.")

    # Map for quick lookup
    prompts_by_id = {int(row.id): row.prompt for _, row in prompts_df.iterrows()}
    inst_by_id = {int(row.id): row for _, row in instances_df.iterrows()}

    for idx in to_process:
        row = pairs_df.loc[idx]
        prompt_id = int(row["prompt_id"])
        instance_id = int(row["instance_id"])

        if prompt_id not in prompts_by_id or instance_id not in inst_by_id:
            print(f"Skipping pair idx={idx}: missing prompt_id={prompt_id} or instance_id={instance_id}.")
            continue

        prompt_template = prompts_by_id[prompt_id]
        inst_row = inst_by_id[instance_id]

        # Parse instance fields
        try:
            numbers = ast.literal_eval(inst_row["numbers"])
        except Exception:
            numbers = inst_row["numbers"]
            if isinstance(numbers, str):
                numbers = list(map(int, re.findall(r"\d+", numbers)))
        target = inst_row["target"]

        # Fill placeholders
        numbers_str = format_numbers_for_prompt(numbers)
        full_prompt = (
            prompt_template
            .replace("<numbers_placeholder>", numbers_str)
            .replace("<target_placeholder>", str(target))
        )

        # Call local Llama (from HF)
        response_text = call_llama_local(
            full_prompt,
            repo_id=args.repo_id,
            filename=args.filename,
            max_tokens=args.max_tokens,
            temperature=args.temp,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            seed=args.seed_local,
            verbose=args.verbose,
        )

        # Evaluate correctness
        correct, message = check_answer(numbers, target, response_text)

        pairs_df.at[idx, "response"] = response_text
        pairs_df.at[idx, "correct"] = int(bool(correct))
        pairs_df.at[idx, "message"] = message

        print(f"Pair idx={idx}: prompt_id={prompt_id}, instance_id={instance_id}, correct={correct}, msg={message}")

        # 🔐 Save after each iteration for safety
        pairs_df.to_csv(args.pairs_file, index=False)

    print(f"Updated results saved incrementally to {args.pairs_file}")


if __name__ == "__main__":
    main()