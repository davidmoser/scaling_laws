# Re-producing Transformer‐Scaling Laws on a Shoestring GPU

This repo recreates **Kaplan et al.**’s “optimal-compute” scaling law
>  *test loss L vs. non-embedding compute C (PF-days) with unlimited data*  

but **only uses sub-10 M-parameter GPT-2 mini-models trained for ~30 GPU-hours**.

[Kaplan, J., McCandlish, S., et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361](https://arxiv.org/abs/2001.08361).

---

## 1 Goal

* **Reproduce** the unlimited-data scaling law (Kaplan et al., 2020): L(C) ∝ C<sup>−0.050</sup>
* **Use tiny models** so the whole study fits in ≈ 30 GPU-hours.
* **Exercise the tooling** — HF Transformers (GPT2), streaming C4, flash-attention 2

---

## 2 Methods

| Element | Setting                                                                |
|---------|------------------------------------------------------------------------|
|Dataset | `allenai/c4` (streaming)                                               |
|Tokeniser | GPT-2 BPE, 50257 vocab                                                 |
|Context | 1024 tokens                                                            |
|Model sweep | d<sub>model</sub> ∈ {32…256}, n<sub>layer</sub>=12, n<sub>head</sub>=8 |
|Batch size | 32 sequences for all runs                                              |
|Optimiser | AdamW, LR = 1 e-3, no weight-decay                                     |
|Accelerator | 1× A100-40 GB, BF16, FlashAttention-2                                  |
|Time | 30 h total, 50 k – 150 k tokens s⁻¹ throughput                         |

<p><b>Non-embedding parameters per Transformer:</b></p>
<p style="margin-left:2em">
N<sub>non-embed</sub> = n<sub>layer</sub> &nbsp;(12&nbsp;d<sub>model</sub><sup>2</sup> + 13&nbsp;d<sub>model</sub>) + 2&nbsp;d<sub>model</sub>
</p>

<p><b>Non-embedding compute (FLOPs):</b></p>
<ul style="margin-left:2em">
  <li>
    per token: &nbsp;
    FLOPs<sub>token</sub> = 2&nbsp;N<sub>non-embed</sub> + 2&nbsp;n<sub>layer</sub> n<sub>ctx</sub> d<sub>model</sub>
  </li>
  <li>
    per training step: &nbsp;
    FLOPs<sub>step</sub> = B &nbsp;n<sub>ctx</sub> &nbsp;FLOPs<sub>token</sub>
    &nbsp; &nbsp; (where&nbsp;B = batch&nbsp;size)
  </li>
</ul>


---

## 3 Results

### 3.1 Loss vs. non-embedding compute

![Test loss vs PF-days](results/loss_vs_pf_days.png)

* Clear **two-regime kink**: fast initial drop, then slower power-law tail.
* Linear fit to the convex-hull segment (after the kink):

<div style="border:1px solid; display:inline-block; padding:0.2em 0.4em;">
  L(C) = (<span style="white-space:nowrap;">C / 6.6×10<sup>4</sup></span>)<sup>−0.0717</sup>
</div>

Kaplan’s efficient-frontier fit (Figure 1): L(C) = (<span style="white-space:nowrap;">C / 2.3×10<sup>8</sup></span>)<sup>−0.050</sup>

### 3.2 Loss vs. tokens processed

![Test loss vs tokens](results/loss_vs_tokens.png)

* Small models learn **slower at every stage**; they are compute-inefficient.
* Identical batch size highlights that width alone drives the differences.
* Compare to Figure 2 in Kaplan et al.

---

## 4 Discussion & comparison to the original paper

* **Same batch size** for every run ⇒ we plot *raw* compute, not the batch-corrected \(C<sub>min</sub>
  \). That explains the steeper exponent (-0.0717 vs -0.050).
* **Width sweep only** (layers held at 12), so architectural degrees of freedom differ from Kaplan et al.’s joint width/depth scaling.
* Absolute loss is lower (≈ 3.8 – 6.3) due to tokenizer and vocabulary differences; the slope comparison is what matters.

---

## 6 Reproduce / extend

1. `git clone https://github.com/davidmoser/scaling_laws.git && cd scaling_laws`
2. `pip install -r requirements.txt`
3. Set `HF_TOKEN` with access to *allenai/c4*, `cd src`
4. `python -m train_model.py` — expect ≈ 30 GPU-hours on one A100.
5. `python -m plot_loss_vs_compute.py` — generate compute plot & fit print-out.
6. `python -m plot_loss_vs_tokens.py` — generate token plots

Feel free to fork, swap schedulers, vary batch size, or push to larger d-models to see how the exponent bends toward the canonical −0.05 line.