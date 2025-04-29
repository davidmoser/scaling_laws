import json

import matplotlib.pyplot as plt
import numpy as np

from src.model_config import TrainingResults

pf_days = True

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

min_loss = 1e20
max_loss = 0
min_compute = 1e20
max_compute = 0

for file in files:
    with open(file, 'r') as f:
        data = json.load(f)

    results = TrainingResults.from_dict(data)

    steps, train_loss = zip(*results.train_loss)
    _, eval_loss = zip(*results.eval_loss)

    steps = np.array(steps)
    compute = results.config.flops_per_step() * steps / (1e15 * 24 * 60 *60 if pf_days else 1)

    min_compute = min(min_compute, compute[10])
    max_compute = max(max_compute, compute[-1])
    min_loss = min(min_loss, eval_loss[-1])
    max_loss = max(max_loss, eval_loss[10])

    plt.loglog(compute[10:], eval_loss[10:], label=f"{results.config.num_parameters(False)/1e6:.1f}m", linewidth=1)

plt.xlabel("PF-days" if pf_days else "Flops")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()

slope = (np.log(min_loss) - np.log(max_loss)) / (np.log(max_compute) - np.log(min_compute))
intercept = np.log(min_loss) - slope * np.log(max_compute)
print(f"slope: {slope:.4f}, intercept: {intercept:.4f}, C_0: {np.exp(-intercept/slope):.4f}")