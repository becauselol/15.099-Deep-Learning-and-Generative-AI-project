import os
import ast
import math
import random
import argparse
import re
import time
from typing import Optional

import pandas as pd

# -----------------------------
# Gemini setup (singleton)
# -----------------------------

_GEMINI_MODEL = None
_GEMINI_MODEL_KEY = None  # (model_variant, temperature, max_tokens)


def _get_gemini_model(
    model_variant: str = "2.5-flash-lite",
    temperature: float = 0.0,
    max_output_tokens: Optional[int] = None,
) :
    """
    Lazy-init a single Gemini GenerativeModel instance.
    Respects GEMINI_API_KEY (or .env).
    """
    global _GEMINI_MODEL, _GEMINI_MODEL_KEY

    key = (model_variant, float(temperature), int(max_output_tokens) if max_output_tokens is not None else None)
    if _GEMINI_MODEL is not None and _GEMINI_MODEL_KEY == key:
        return _GEMINI_MODEL

    import google.generativeai as genai
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment or .env file.")

    genai.configure(api_key=api_key)

    generation_config = {"temperature": float(temperature)}
    if max_output_tokens is not None:
        generation_config["max_output_tokens"] = int(max_output_tokens)

    model_name = f"gemini-{model_variant}"
    _GEMINI_MODEL = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        # system_instruction=None  # you could add a system prompt here if desired
    )
    _GEMINI_MODEL_KEY = key
    return _GEMINI_MODEL


def call_gemini(
    prompt: str,
    *,
    model_variant: str = "2.5-flash-lite",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    retries: int = 3,
    backoff_s: int = 2,
    sleep_time: float = 0.0,
) -> str:
    """
    Call Gemini API to get a response to the given prompt.
    Handles retries and basic response parsing.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            model = _get_gemini_model(
                model_variant=model_variant,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            if sleep_time > 0:
                time.sleep(sleep_time)

            response = model.generate_content(prompt)

            # Primary text
            text = getattr(response, "text", None)
            if not text:
                # Fallback: stitch from candidates/parts if needed
                candidates = getattr(response, "candidates", []) or []
                if candidates and getattr(candidates[0], "content", None):
                    parts = getattr(candidates[0].content, "parts", []) or []
                    text = "".join(getattr(p, "text", "") for p in parts if hasattr(p, "text"))
            if not text:
                raise RuntimeError("Empty response from Gemini.")

            return text
        except Exception as e:
            last_err = e
            if attempt == retries:
                raise RuntimeError(f"Gemini call failed after {retries} attempts: {e}") from e
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
    - Replaces x/× with * for multiplication.
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
    # Normalize common multiplication symbols
    answer_content = answer_content.replace("x", "*").replace("×", "*")
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
    parser = argparse.ArgumentParser(description="Run Gemini on random (prompt, instance) Countdown pairs.")
    parser.add_argument("--prompts_file", type=str, default="countdown_prompts_gemini_transformed.csv")
    parser.add_argument("--instances_file", type=str, default="countdown_features.csv")
    parser.add_argument("--pairs_file", type=str, default="countdown_results_gemini.csv")
    parser.add_argument("--num_pairs", type=int, default=1000,
                        help="Number of random prompt-instance pairs to generate if pairs_file does not exist.")
    parser.add_argument("--max_calls", type=int, default=1000,
                        help="Maximum number of new LLM calls to make in this run.")

    # Gemini params
    parser.add_argument("--model_variant", type=str, default="2.5-flash-lite",
                        help="Gemini model variant, e.g. '2.0-flash', '2.5-flash-lite'.")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--sleep_time", type=float, default=4.0,
                        help="Optional sleep between Gemini calls (seconds), e.g. for rate limiting.")

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

        # Call Gemini
        response_text = call_gemini(
            full_prompt,
            model_variant=args.model_variant,
            max_tokens=args.max_tokens,
            temperature=args.temp,
            sleep_time=args.sleep_time,
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