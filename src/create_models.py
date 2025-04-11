"""
Script for creating language models at different total (non-embedding)
parameter scales, then pushing them to the Hugging Face Hub.

n_vocab: from tokenizer
d_model: residual dimension
n_heads: number of attention heads
d_attn: dimension of key, query and value space (split over n_heads)
d_ff: dimension of feed forward space

parameters:
E: n_vocab x d_model
W^Q, W^K, W^V, W^O: 4 x d_model x d_attn x n_layer
F_1, F_2: 2 x d_model x d_ff x n_layer

Use d_attn = d_model = d_ff / 4

Total parameters: 12 x n_layer x d_model^2
"""

import os

from huggingface_hub import login
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

# Number of non-embedding parameters in the models
D_MODELS = [16, 32, 64, 128, 256, 512, 1024]
N_LAYER = 6
N_HEAD = 4
HF_REPO = "gebregl"


def count_nonembedding_params(model):
    """
    Count total parameters minus embedding params.
    For GPT2LMHeadModel, embedding modules typically:
      - model.transformer.wte (word embeddings)
      - model.transformer.wpe (positional embeddings)
    """
    total_params = sum(p.numel() for p in model.parameters())

    embedding_params = 0
    if hasattr(model.transformer, 'wte'):
        embedding_params += sum(p.numel() for p in model.transformer.wte.parameters())
    if hasattr(model.transformer, 'wpe'):
        embedding_params += sum(p.numel() for p in model.transformer.wpe.parameters())

    return total_params - embedding_params


def create_and_push_model(d_model, n_layer, n_head):
    config = GPT2Config(
        n_embd=d_model,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * d_model,  # d_ff
        vocab_size=50257,  # GPT-2 default
    )
    model = GPT2LMHeadModel(config)
    model_name = f"llm-scaling-{d_model}"

    repo_id = f"{HF_REPO}/{model_name}"
    print(f"Pushing {repo_id} to hub...")
    model.push_to_hub(repo_id=repo_id)


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Log in to huggingface-hub (requires prior HF CLI login or token env var)
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)

    for d_model in D_MODELS:
        create_and_push_model(d_model, n_layer=N_LAYER, n_head=N_HEAD)


if __name__ == "__main__":
    main()
