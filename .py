# 1. Installing the environment
!pip install -q transformers accelerate pandas numpy scipy matplotlib torch
# 2. Importing & Config required files
import re
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

TOKEN_BUDGETS = [10, 20, 40, 160, 240, 5000]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device:", DEVICE)
# 3. Creating a dataset consisting of 3 questions and answers
DATASET = [
    {
        "prompt_id": "P01",
        "question": (
            "Alice has 3 boxes. Each box contains 4 apples. "
            "She gives 5 apples to Bob. How many apples does Alice have left?"
        ),
        "ground_truth_answer": "7",
    },
    {
        "prompt_id": "P02",
        "question": (
            "A train travels 60 km per hour for 5 hours. "
            "After the trip, it travels 40 km less because of a route change. "
            "What is the final distance travelled?"
        ),
        "ground_truth_answer": "260",
    },
    {
        "prompt_id": "P03",
        "question": (
            "Sarah earns $14 per hour and works 40 hours per week. "
            "She spends $60 on transportation each week. "
            "How much money does she keep after transportation?"
        ),
        "ground_truth_answer": "500",
    },
]
# 4. Load Model and tokenize
def load_model(model_name=MODEL_NAME):
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    return tokenizer, model

TOKENIZER, MODEL = load_model()

# 5. Prompting & Extraction from the model
SYSTEM_PROMPT = (
    "You are a careful mathematics assistant. "
    "Provide brief reasoning before the final answer. "
    "Use this exact format:\n"
    "Reasoning: <brief reasoning>\n"
    "Answer: <final numeric answer>\n"
    "Do not include units in the final answer."
)

def build_prompt(question: str, tokenizer=TOKENIZER) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question:\n{question}"},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    return f"System: {SYSTEM_PROMPT}\n\nUser: Question:\n{question}\n\nAssistant:"


def extract_answer(text: str) -> str:
    """
    Extract numeric answer after 'Answer:'.
    Fallback: last number in generated text.
    Handles integers, decimals, negatives, commas.
    """
    match = re.search(
        r"Answer:\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).replace(",", "").strip()

    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else ""


def extract_reasoning(text: str) -> str:
    """
    Extract text between Reasoning: and Answer: if available.
    """
    match = re.search(
        r"Reasoning:\s*(.*?)(?:Answer:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return ""


def is_correct(pred: str, truth: str) -> bool:
    try:
        return abs(float(pred) - float(truth)) < 1e-6
    except Exception:
        return pred.strip() == truth.strip()


def has_reasoning_marker(text: str) -> bool:
    return bool(re.search(r"Reasoning:", text, flags=re.IGNORECASE))

# 6. Generation & Logit Utilities with budget

def generate_with_budget(prompt: str, max_new_tokens: int):
    """
    Deterministic generation under a fixed output token budget.
    Returns generated text, output token count, runtime, final logits.
    """
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    input_len = inputs["input_ids"].shape[1]

    start = time.time()

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=TOKENIZER.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    duration = time.time() - start

    generated_ids = outputs.sequences[0][input_len:]
    generated_text = TOKENIZER.decode(generated_ids, skip_special_tokens=True)

    output_token_count = len(generated_ids)

    # Last-step logits distribution if available
    if outputs.scores:
        final_logits = outputs.scores[-1][0].detach().float().cpu()
    else:
        final_logits = None

    return {
        "generated_text": generated_text,
        "output_tokens": output_token_count,
        "duration_sec": duration,
        "final_logits": final_logits,
    }


def answer_token_probability(final_logits, answer: str) -> float:
    """
    Probability assigned to the first token of the ground-truth answer.
    """
    if final_logits is None or answer == "":
        return np.nan

    answer_ids = TOKENIZER.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        return np.nan

    probs = F.softmax(final_logits, dim=-1)
    return probs[answer_ids[0]].item()


def kl_divergence(logits_p, logits_q) -> float:
    """
    KL(P || Q), where P and Q are final-token distributions.
    """
    if logits_p is None or logits_q is None:
        return np.nan

    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = torch.exp(log_p)

    return (p * (log_p - log_q)).sum().item()

# 7. Main Experiment and run budget

def run_budget_experiment(dataset=DATASET, token_budgets=TOKEN_BUDGETS):

    all_rows = []

    for ex in dataset:

        pid = ex["prompt_id"]
        question = ex["question"]
        truth = ex["ground_truth_answer"]

        print("\n" + "=" * 70)
        print(f"Prompt {pid}: {question[:80]}...")
        print("=" * 70)

        prompt = build_prompt(question)

        per_budget_outputs = {}

        for budget in token_budgets:

            result = generate_with_budget(
                prompt=prompt,
                max_new_tokens=budget
            )

            text = result["generated_text"]

            answer = extract_answer(text)
            reasoning = extract_reasoning(text)

            row = {
                "Prompt_ID": pid,
                "Question": question,
                "Budget": budget,
                "Output_Tokens": result["output_tokens"],
                "Duration_Sec": result["duration_sec"],
                "Generated_Text": text,
                "Generated_Answer": answer,
                "Ground_Truth": truth,
                "Correctness": int(is_correct(answer, truth)),
                "Answer_Prob": answer_token_probability(
                    result["final_logits"],
                    truth
                ),
                "Final_Logits": result["final_logits"],
            }

            per_budget_outputs[budget] = row

            print(
                f"Budget={budget:>3} | "
                f"tokens={row['Output_Tokens']:>3} | "
                f"answer={answer:>8} | "
                f"correct={'✓' if row['Correctness'] else '✗'} | "
            )

            print("\n--- GENERATED OUTPUT ---")
            print(text)
            print("-" * 50)

        # Largest budget as baseline
        full_budget = max(token_budgets)
        full_logits = per_budget_outputs[full_budget]["Final_Logits"]

        for budget, row in per_budget_outputs.items():

            row["KL_vs_FullBudget"] = (
                0.0 if budget == full_budget
                else kl_divergence(full_logits, row["Final_Logits"])
            )

            row.pop("Final_Logits", None)

            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    return df


results_df = run_budget_experiment()

# 8. Summary with tables

summary_df = (
    results_df
    .groupby("Budget")
    .agg(
        Accuracy=("Correctness", "mean"),
        Mean_Output_Tokens=("Output_Tokens", "mean"),
        Mean_KL_vs_FullBudget=("KL_vs_FullBudget", "mean"),
    )
    .reset_index()
)

print("=== Summary by Token Budget ===")
display(summary_df)

print("=== Full Results ===")
display(results_df[[
    "Prompt_ID",
    "Budget",
    "Generated_Answer",
    "Ground_Truth",
    "Correctness",
    "Output_Tokens",
    "KL_vs_FullBudget",
]])

# 9. Plots for analysis

plt.figure()
plt.plot(summary_df["Budget"], summary_df["Accuracy"], marker="o")
plt.xlabel("Max New Tokens")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Generation Budget")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(summary_df["Budget"], summary_df["Mean_KL_vs_FullBudget"], marker="o")
plt.xlabel("Max New Tokens")
plt.ylabel("Mean KL vs Full Budget")
plt.title("Distribution Shift vs Full Budget")
plt.grid(True)
plt.show()

# 10. Saving the results if needed

#results_df.to_csv("cot_budget_full_results.csv", index=False)
#summary_df.to_csv("cot_budget_summary.csv", index=False)

#print("Saved:")
#print("cot_budget_full_results.csv")
#print("cot_budget_summary.csv")
