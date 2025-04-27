import json

import matplotlib.pyplot as plt

from src.model_config import TrainingResults

with open('../results/gpt2_64_results.json', 'r') as f:
    data = json.load(f)

results = TrainingResults.from_dict(data)

train_steps, train_loss = zip(*results.train_loss)
eval_steps, eval_loss = zip(*results.eval_loss)

# Plot
plt.figure()
plt.loglog(train_steps, train_loss, label="Training loss", linewidth=1)
plt.loglog(eval_steps, eval_loss, label="Validation loss", linewidth=1)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()
