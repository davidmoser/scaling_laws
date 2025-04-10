"""
Example script for creating small GPT2-like models at different total (non-embedding)
parameter scales, then pushing them to the Hugging Face Hub.

Requirements:
    pip install transformers huggingface-hub
    # Also ensure you're logged in to your HF account:
    huggingface-cli login
"""

import math
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    HfApi,
    HfFolder
)

# Example target sizes for NON-EMBEDDING params, e.g. 1e3, 1e4, 1e5, etc.
# Adjust this list (and shape logic) to your needs/hardware.
TARGET_NONEMBED_SIZES = [1e2, 1e3, 1e4, 1e5, 1e6]

# We choose a fixed number of layers. We'll vary n_embd to scale the model.
N_LAYERS = 6

# For GPT-2, there's a well-known approximate formula for non-embedding params:
#   N_nonembed ~ 12 * (n_layer) * (n_embd^2)
# We'll invert this to solve for n_embd given N_nonembed, rounding to an integer.
def get_n_embd_from_nonembed(N_nonembed, n_layer):
    # Just invert the approximate formula from the scaling laws for GPT-style.
    # N_nonembed = 12 * n_layer * (n_embd^2)
    base = N_nonembed / (12 * n_layer)
    # Take sqrt, then floor it so we don't overshoot the target.
    n_embd = int(math.sqrt(base))
    return max(n_embd, 1)  # to avoid zero or negative in extreme cases

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

def create_and_push_model(target_nonembed, n_layer=N_LAYERS):
    # 1. Solve for n_embd:
    n_embd = get_n_embd_from_nonembed(target_nonembed, n_layer)

    # 2. Decide on a suitable n_head. For GPT-2 usually n_head divides n_embd evenly.
    #    We'll pick a small integer factor if possible, else default to 1 for tiny models.
    #    This is a simplistic approach. Adjust as needed.
    possible_heads = [1, 2, 4, 8, 16, 32]
    n_head = max(h for h in possible_heads if h <= n_embd and (n_embd % h == 0)) if n_embd > 0 else 1

    # 3. Create config
    config = GPT2Config(
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        # Typically GPT-2 sets feed-forward dim as 4*n_embd, but it's "n_inner" in HF config
        n_inner=4 * n_embd,
        vocab_size=50257,  # GPT-2 default
        # you can also adjust other settings like max_position_embeddings
    )

    # 4. Instantiate model
    model = GPT2LMHeadModel(config)

    # 5. Compute actual non-embedding params
    actual_nonembed = count_nonembedding_params(model)
    log10_nonembed = round(math.log10(actual_nonembed), 2) if actual_nonembed > 0 else 0.0

    # 6. Give the model a name that encodes log10 of # non-embed params
    # E.g. "gpt2-minimal-nonembed-4.30" => ~1.995e4 non-embed
    model_name = f"gpt2-mini-nonembed-{log10_nonembed:.2f}"

    print(f"Created model {model_name} with approx. {actual_nonembed} non-embed params.")

    # 7. Push to Hugging Face Hub (must be logged in)
    #    Adjust 'repo_id' to your user or org handle.
    repo_id = f"<YOUR_USERNAME_OR_ORG>/{model_name}"
    print(f"Pushing {repo_id} to hub...")
    model.push_to_hub(repo_id=repo_id)

    # (Optional) push config separately if you want
    config.push_to_hub(repo_id=repo_id)

def main():
    """
    Create and push multiple GPT-2–style models of increasing size.
    """
    # Just initialize the GPT-2 tokenizer once (downstream training might reuse it)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Log in to huggingface-hub (requires prior HF CLI login or token env var)
    #   - If using a token, do: HfFolder.save_token(<YOUR_TOKEN>)
    hf_api = HfApi()
    # Confirm we have a token
    token = HfFolder.get_token()
    if token is None:
        raise ValueError("No HF token found. Please login via `huggingface-cli login` or set HF_TOKEN.")

    # Loop through desired target sizes
    for target_size in TARGET_NONEMBED_SIZES:
        create_and_push_model(target_size, n_layer=N_LAYERS)

if __name__ == "__main__":
    main()
