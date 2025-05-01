# Scaling-laws-mini – re-deriving Kaplan et al. with pocket-sized models

> *Reproduce the “optimal loss vs. non-embedding compute” power-law using only sub-million-parameter GPT-style models on a single A100.*

---

## 1 Goal

* **Reproduce** the unlimited-data scaling law  
  \(L(C_\text{min})\propto C_\text{min}^{-0.05}\) (Kaplan et al., 2020)
* **Use tiny models** so the whole study fits in ≈ 30 GPU-hours.
* **Exercise the tooling** — HF Transformers, streaming C4, flash-attention 2 — and store raw logs for further analysis.

---

## 2 Method (`src/`)

| Element | Setting                                                                                    |
|---------|--------------------------------------------------------------------------------------------|
|Dataset | `allenai/c4` (streaming)                                                                   |
|Tokeniser | GPT-2 BPE, 50257 vocab                                                                     |
|Context | 1024 tokens                                                                                |
|Model sweep | 9 runs, **d<sub>model</sub> ∈ {32…256}**, **n<sub>layer</sub>=12**, **n<sub>head</sub>=8** |
|Batch size | **32** sequences for *all* runs                                                            |
|Optimiser | AdamW, LR = 1 e-3, no weight-decay                                                         |
|Accelerator | 1× A100-40 GB, BF16, FlashAttention-2                                                      |
|Time | 30 h total ⇒ **50 k – 150 k tokens s⁻¹** throughput                                        |

Compute is reported in **PF-days**  
\(C = \text{steps} × \text{FLOPs/step} / (10^{15}⋅86400)\).

---

## 3 Results

### 3.1 Loss vs. non-embedding compute

![Test loss vs PF-days](results/loss_vs_pf_days.png)

* Clear **two-regime kink**: fast initial drop, then slower power-law tail.
* Linear fit to the convex-hull segment (after the kink):

\[
\boxed{L(C)=\left(\dfrac{C}{6.6\times10^{4}}\right)^{-0.0717}}
\]

Kaplan’s efficient-frontier fit: \(L(C)=(C/2.3\times10^{8})^{-0.050}\).

### 3.2 Loss vs. tokens processed

![Test loss vs tokens](results/loss_vs_tokens.png)

* Small models learn **slower at every stage**; they are compute-inefficient.
* Identical batch size highlights that width alone drives the differences.

---

## 4 Discussion & deviations from the original paper

* **Same batch size** for every run ⇒ we plot *raw* compute, not the batch-corrected \(C_\text{min}\). That explains the steeper exponent (-0.0717 vs -0.050).
* **Width sweep only** (layers held at 12), so architectural degrees of freedom differ from Kaplan et al.’s joint width/depth scaling.
* Absolute loss is lower (≈ 3.8 – 6.3) due to tokenizer and vocabulary differences; the slope comparison is what matters.

---

## 6 Reproduce / extend

1. `git clone https://github.com/davidmoser/scaling_laws.git && cd scaling_laws`
2. `pip install -r requirements.txt` (Tor​ch 2.7.0, Transformers 4.51.1, …)
3. Set `HF_TOKEN` with access to *allenai/c4*.
4. `python -m src.train` — expect ≈ 30 GPU-hours on one A100.
5. `python -m src.plot` — regenerate plots & fit print-out.

Feel free to fork, swap schedulers, vary batch size, or push to larger d-models to see how the exponent bends toward the canonical −0.05 line.