import pandas as pd
import json
import os
import time
import csv
from dataclasses import dataclass
from typing import List
from llama_cpp import Llama
import re

@dataclass
class Prompt:
    text: str
    features: List[str]

@dataclass
class CountdownInstance:
    numbers: List[int]
    target: int
    solution: str = ""

class PromptGenerator:
    def __init__(self):
        self.llm_singleton = None

    def generate_instruction_prompt_template(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    
    def generate_alternative_prompts_prompt(self, instruction_prompt, num_alternatives=5):
        prompt = f"Given an original prompt, create exactly {num_alternatives} alternative prompts that preserve the same underlying task as the original prompt, but may vary phrasing, structure, and additional guidance. Do not change what the original prompt is ultimately asking the model to do.\n"
        prompt += """
You must output ONLY a JSON array, with no extra text. The structure is:
[
  {
    "prompt": "<alternative prompt text 1>",
    "features": [
      <list of feature labels describing the transformations applied>
    ]
  },
  {
    "prompt": "<alternative prompt text 2>",
    "features": [
      <list of feature labels describing the transformations applied>
    ]
  }, ...
]\n"""
        prompt += f"""where "..." is an illustration only and indicates continuation of the pattern until there are {num_alternatives} total alternative prompts.\n"""
        prompt += """
Hard requirements for the JSON:
 - The output MUST be a single valid JSON array.
 - Each element MUST be a JSON object with exactly two keys: "prompt" (string) and "features" (array of strings).
 - The "features" array MUST contain at least one label.
 - Do NOT include comments, ellipses (…), or trailing commas.
 - Do NOT include the original prompt text anywhere in the JSON output.
 - All placeholder tokens from the original prompt (e.g., <numbers_placeholder>, <target_placeholder>) MUST appear unchanged in every alternative prompt.
 - All formatting instructions from the original prompt (such as <answer></answer> and <think></think> tags) MUST be preserved exactly in every alternative prompt.
 - Before returning the output, you MUST silently verify that it is VALID JSON and conforms to all STRUCTURAL and FORMATTING rules. If it is not valid, you MUST correct it automatically.
 - The final output MUST contain ONLY the corrected JSON array, with no additional commentary.

For each alternative prompt, you MUST:
- Identify which prompting strategies you used.
- Encode each strategy as a feature label in "features". If multiple strategies are used in a single prompt, include ALL relevant labels.
- Use ONLY the labels defined below.
- If the only change from the original prompt is rephrasing while keeping the same task, you MUST include "paraphrasing" in "features".

You MUST interpret and use the following feature labels exactly as defined:

1. "paraphrasing"
   - Meaning: The instruction is semantically equivalent to the original prompt, but reworded (changed phrasing, synonyms, sentence structure) without intentionally changing the task.
   - Example change: Reordering clauses, replacing words with synonyms, simplifying or slightly rephrasing sentences.

2. "role-specification"
   - Meaning: The prompt explicitly assigns a role or identity to the model (e.g., "You are a medical doctor", "You are a helpful assistant").
   - Example change: Adding or modifying a sentence that specifies who the model is or how it should behave.

3. "reasoning-trigger"
   - Meaning: The prompt explicitly asks for step-by-step thinking or reasoning before giving the final answer.
   - Example phrases: "Think step by step", "Explain your reasoning first, then give the answer", "Show your work".

4. "chain-of-thought"
   - Meaning: A specific type of reasoning-trigger where the model is explicitly instructed to produce a detailed, multi-step explanation of its thought process.
   - Example phrase: "First reason about the problem in detail, then provide the final answer."

5. "self-check"
   - Meaning: The prompt explicitly asks the model to verify, critique, or check its own answer for correctness, consistency, or completeness.
   - Example phrase: "After you produce an answer, check for errors and correct them if necessary."

6. "<k>-shot" (e.g., "1-shot", "2-shot", "5-shot")
   - Meaning: The prompt includes exactly k > 0 demonstration examples (input + output pairs) before the actual query.
   - Replace <k> with the actual positive integer number of examples (e.g., "1-shot", "2-shot").
   - This label MUST always be consistent with the actual number of examples in the prompt.
   - Use a "<k>-shot" label ONLY if such examples are actually present.

7. "conciseness"
   - Meaning: The prompt is intentionally kept short and minimal while still being understandable. It avoids unnecessary detail or verbosity.
   - Example change: Removing extra explanations, disclaimers, or long context that are not strictly needed.

8. "verbosity"
   - Meaning: The prompt is intentionally more detailed or elaborate, adding explanations, clarifications, or extra context to guide the model.
   - Example change: Expanding instructions, describing the setting, or giving more detailed constraints.

9. "context-expansion"
    - Meaning: The prompt adds background information, assumptions, or scenario details that were not explicitly present in the original prompt, to help the model understand the task better.
    - Example change: Adding a brief scenario, definitions, or domain context.

Example:
Original prompt: "Using the numbers <numbers_placeholder>, create an equation that equals <target_placeholder>. You can use basic arithmetic operations (+, -, *, /) and each number MUST BE used EXACTLY once. Enclose only the final equation within <answer></answer> tags. If intermediate reasoning or derivations are needed, place them inside <think></think> tags. Do not include any text or explanation outside these tags."
LLM output for 2 alternatives (starts in the next line):
[
  {
    "prompt": "Create an equation using <numbers_placeholder> that results in <target_placeholder>. Use only addition (+), subtraction (-), multiplication (*), and division (/), and ensure each number is used exactly once. Output the final equation inside <answer></answer>, and put any intermediate reasoning or derivations inside <think></think>.",
    "features": [
      "paraphrasing",
      "conciseness"
    ]
  },
  {
    "prompt": "Your task is to carefully construct a mathematically valid equation that evaluates exactly to the value <target_placeholder>, making use of every element contained within <numbers_placeholder> exactly once and only once, without exception. You are allowed to apply only the basic arithmetic operations — specifically addition (+), subtraction (-), multiplication (*), and division (/). In performing this task, you should approach the problem as if you are demonstrating your full thought process to a student observing your reasoning. Therefore, you must break down your solution into a clear, multi-stage explanation. Begin by considering all available numbers and all possible pairings or groupings among them. Then explore several plausible combinations of arithmetic operations that could be applied to move progressively closer to the target value. As you carry out this exploration, write every intermediate calculation and logical inference inside <think></think> tags so that the structure of your reasoning is fully transparent. After you have iteratively refined your expression and confirmed that it evaluates exactly to <target_placeholder>, present the final, simplified equation within <answer></answer> tags. The final equation must appear only inside <answer></answer> and must not include any surrounding commentary, justification, or explanation, which must instead remain solely inside the <think></think> tags.",
    "features": [
        "paraphrasing",
        "verbosity",
        "reasoning-trigger",
        "chain-of-thought",
        "tone-adjustment"
    ]
  }
]
"""        
        prompt += f"\nOriginal prompt: {instruction_prompt}\n"
        return prompt

    def load_instances(self, json_file_path: str):
        """Load instances from a JSON file into appropriate dataclass objects."""
        raise NotImplementedError("Subclasses must implement this method")

    def save_prompts(self, prompts, output_file):
        """Save generated prompts to the prompts file."""
        file_exists = os.path.exists(output_file)

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["features", "prompt"])

            for prompt in prompts:
                writer.writerow([json.dumps(prompt.features), prompt.text])

    def _get_llm(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1, seed: int = 0, verbose: bool = False) -> Llama:
        """
        Lazy-init a single Llama instance. n_gpu_layers=-1 => offload all to GPU (CUDA build).
        """
        if self.llm_singleton is None:
            # self.llm_singleton = Llama(
            #     model_path=os.path.abspath(os.path.expanduser(model_path)),
            #     n_ctx=int(n_ctx),
            #     n_gpu_layers=int(n_gpu_layers),
            #     seed=int(seed),
            #     logits_all=False,
            #     verbose=bool(verbose)
            # )
            self.init_gemini()

        return self.llm_singleton

    def init_gemini(self, model_variant="2.5-flash", api_key=None):
        """
        Initialize a Gemini client/model handle.
        """
        import google.generativeai as genai

        # Configure API key
        from os import getenv
        from dotenv import load_dotenv
        load_dotenv()
        key = api_key or getenv("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set and no api_key was provided.")
        genai.configure(api_key=key)

        self.model_id = "gemini-" + model_variant

        # Build the model with the system instruction baked in
        sys_text = self.system.render() if hasattr(self, "system") and self.system else ""
        self.llm_singleton = genai.GenerativeModel(
            model_name=self.model_id,
            system_instruction=sys_text or None,
            generation_config={
                "temperature": 0.7,
                # Set this if you want a hard cap:
                # "max_output_tokens": 2048,
            },
            # Optional: add safety settings if your app needs them
            # safety_settings=[...]
        )

    def call_gemini(
        self,
        prompt: str,
        model_variant="2.5-flash",
        use_quantization=False,
        sleep_time=0.0,
    ) -> str:
        """Call Gemini API to get a response to the given prompt."""

        # Lazy init if needed
        if not getattr(self, "llm_singleton", None):
            self.init_gemini(model_variant=model_variant)

        time.sleep(sleep_time)

        # Generate
        try:
            # With system instruction already set on the model, we only pass the user turn.
            # If you prefer explicit roles, you can also pass a list of parts:
            # response = self.llm_singleton.generate_content([{"role": "user", "parts": [self.user_prompt]}])
            response = self.llm_singleton.generate_content(prompt)

            # Gemini responses commonly expose .text for the primary candidate.
            text = getattr(response, "text", None)
            if not text:
                # Fallback: stitch parts if needed
                candidates = getattr(response, "candidates", []) or []
                if candidates and getattr(candidates[0], "content", None):
                    parts = getattr(candidates[0].content, "parts", []) or []
                    text = "".join(getattr(p, "text", "") for p in parts if hasattr(p, "text"))
            if not text:
                raise RuntimeError("Empty response from Gemini.")

            return text

        except Exception as e:
            # Surface a clean error with context
            raise RuntimeError(f"Gemini call failed: {e}") from e

    def call_llama_local(self, prompt: str, *, model_path: str = "/home/lpontes/models/llama-2-7b-chat.Q4_K_M.gguf", max_tokens: int = 1024, temperature: float = 0.7, n_ctx: int = 4096, n_gpu_layers: int = -1, seed: int = 0, retries: int = 3, backoff_s: int = 2, verbose: bool = False) -> str:
        """
        Local GPU chat using llama_cpp on a compute node.
        """
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                llm = self._get_llm(model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, seed=seed, verbose=verbose)
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

    def parse_alternative_prompts(self, llm_output: str):
        """
        Parse messy LLM output that *tries* to be a JSON array of:
        { "prompt": "...", "features": ["...", ...] }

        but may:
        - have extra text before/after
        - miss commas between feature strings
        - contain multiple independent arrays

        We ignore the top-level JSON structure and instead:
        - find each object containing both "prompt" and "features"
        - extract the prompt text
        - extract all quoted strings inside the features [...] block
        """

        results = []

        # Find all object-like chunks that contain both "prompt" and "features"
        # This is fairly generous but works well for your logged outputs.
        object_pattern = re.compile(
            r'\{[^{}]*"prompt"\s*:\s*"[^"]*"[^{}]*"features"\s*:\s*\[[^\]]*\][^{}]*\}',
            re.DOTALL
        )

        for match in object_pattern.finditer(llm_output):
            block = match.group(0)

            # 1. Extract "prompt": "..."
            m_prompt = re.search(r'"prompt"\s*:\s*"([^"]*)"', block, re.DOTALL)
            if not m_prompt:
                continue
            prompt_text = m_prompt.group(1)

            # 2. Extract the content of "features": [ ... ]
            m_features = re.search(r'"features"\s*:\s*\[([^\]]*)\]', block, re.DOTALL)
            if not m_features:
                continue
            features_block = m_features.group(1)

            # 3. Inside that block, extract *all* quoted strings as feature labels.
            # This works even if commas are missing or weirdly formatted.
            features = re.findall(r'"([^"]+)"', features_block)
            features = [f.strip() for f in features if f.strip()]

            if not features:
                # If no features, skip this object
                continue

            results.append(
                {
                    "prompt": prompt_text,
                    "features": features,
                }
            )

        # Optional: sanity check and debugging
        if not results:
            print("parse_alternative_prompts: no valid prompt/features blocks found.")
            print("Raw output snippet:")
            print(llm_output[:500])

        return results
            
    def generate_and_save_alternative_prompts(self, instruction_prompt, output_file, num_alternatives=5):
        llm_prompt = self.generate_alternative_prompts_prompt(instruction_prompt, num_alternatives)
        # llm_output = self.call_llama_local(llm_prompt)
        llm_output = self.call_gemini(llm_prompt, sleep_time=4.0)
        alternatives = self.parse_alternative_prompts(llm_output)
        prompts = []
        for alt in alternatives:
            prompts.append(Prompt(text=alt["prompt"], features=alt["features"]))
        self.save_prompts(prompts=prompts, output_file=output_file)


class CountdownPromptGenerator(PromptGenerator):
    def __init__(self):
        super().__init__()
        self.instruction_prompt = self.generate_instruction_prompt_template()

    def generate_instruction_prompt_template(self):
        instruction = "Using the numbers <numbers_placeholder>, create an equation that equals <target_placeholder>. You can use basic arithmetic operations (+, -, *, /) and each number MUST BE used EXACTLY once. "
        answer_format = "Enclose only the final equation within <answer></answer> tags. If intermediate reasoning or derivations are needed, place them inside <think></think> tags. Do not include any text or explanation outside these tags.\n"
        prompt = f"{instruction}{answer_format}"
        return prompt
    
    def generate_and_save_alternative_prompts(self, output_file, num_alternatives=5):
        super().generate_and_save_alternative_prompts(
            instruction_prompt=self.instruction_prompt,
            output_file=output_file,
            num_alternatives=num_alternatives
        )

    def load_instances(self, json_file_path: str = "countdown.json") -> List[CountdownInstance]:
        """Load countdown instances from a JSON file into CountdownInstance objects."""
        with open(json_file_path, "r") as f:
            data = json.load(f)

        instances = []
        for obj in data:
            instance = CountdownInstance(
                numbers=obj["numbers"],
                target=int(obj["target"]),
                solution=obj.get("solution", "")
            )
            instances.append(instance)

        return instances

if __name__ == "__main__":
    countdown_prompter = CountdownPromptGenerator()
    for i in range(800):
        print(f"Iteration {i+1}")
        countdown_prompter.generate_and_save_alternative_prompts(output_file="countdown_prompts_gemini.csv")
