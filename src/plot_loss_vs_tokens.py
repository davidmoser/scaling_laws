import json

import matplotlib.pyplot as plt
import numpy as np

from src.model_config import TrainingResults

files = [
    '../results/gpt2_32_results.json',
    '../results/gpt2_48_results.json',
    '../results/gpt2_64_results.json',
    '../results/gpt2_96_results.json',
    '../results/gpt2_112_results.json',
    '../results/gpt2_128_results.json',
    '../results/gpt2_160_results.json',
    '../results/gpt2_196_results.json',
    '../results/gpt2_256_results.json',
]


plt.figure()

for file in files:
    with open(file, 'r') as f:
        data = json.load(f)

    results = TrainingResults.from_dict(data)

    steps, train_loss = zip(*results.train_loss)
    _, eval_loss = zip(*results.eval_loss)

    steps = np.array(steps)
    tokens = steps * results.config.n_positions * results.config.batch_size

    plt.semilogx(tokens, eval_loss, label=f"{results.config.num_parameters(False)/1e6:.1f}m", linewidth=1)

plt.xlabel("Tokens")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()
