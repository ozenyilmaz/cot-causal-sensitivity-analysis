# Horizon-Truncation Probes: Are CoT Tokens Causally Load-Bearing?

> A minimal mechanistic interpretability experiment testing whether Chain-of-Thought tokens actively drive answer formation or simply explain a decision already made.

---

## The Core Question

When a language model generates a chain-of-thought, do those intermediate tokens *cause* the final answer — or does the model already "know" the answer early on, and just use CoT tokens to fill in an explanation afterward?

This repository operationalizes that question as a controlled perturbation experiment.

---

## Theoretical Background

There are two competing hypotheses:

**Causal Load-Bearing:** CoT tokens function as external working memory. Each token incrementally refines the answer distribution. Cutting the generation short should degrade accuracy and produce measurable distributional drift.

**Post-Hoc Rationalization:** The answer distribution resolves early — driven by the problem context, not the accumulated reasoning tokens. Subsequent tokens are formatting convention. Truncating them should produce little measurable effect.

---

## Experimental Design

The script runs `Qwen/Qwen2.5-Math-1.5B-Instruct` under **greedy decoding** (`do_sample=False`). Greedy decoding is essential: it makes the generation trajectory deterministic, so the only variable across runs is how many tokens the model is allowed to produce before we cut it off.

**Token budgets:** `[10, 20, 40, 160, 240, 5000]`

For each budget $B$:
1. Generate up to $B$ tokens (the CoT prefix).
2. Run a forward pass on that prefix.
3. Extract the next-token logits at the final position → distribution $Q_B$.
4. Compare $Q_B$ against the unconstrained baseline distribution $P$ (budget = 5000).
