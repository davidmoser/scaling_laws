import json
import matplotlib.pyplot as plt
from pathlib import Path

# Placeholder filename
file_path = Path("../results/gpt2_64_results.json")

# Load the JSON data from the file
with open(file_path, "r") as f:
    data = json.load(f)

# Extract training and evaluation loss data
train_data = data["results"]["train_loss"]
eval_data = data["results"]["eval_loss"]

# Separate steps and losses
train_steps, train_loss = zip(*train_data)
eval_steps, eval_loss = zip(*eval_data)

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
