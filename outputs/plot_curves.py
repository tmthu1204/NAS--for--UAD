import json
import matplotlib.pyplot as plt

# Load results
with open("outputs/results.json", "r") as f:
    res = json.load(f)

roc = res["roc"]

plt.figure(figsize=(7,6))
for cls in roc["fpr"].keys():
    fpr = roc["fpr"][cls]
    tpr = roc["tpr"][cls]
    plt.plot(fpr, tpr, label=f"Class {cls}")

# Đường chéo baseline
plt.plot([0, 1], [0, 1], "k--", label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (per class)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("outputs/roc_curve.png", dpi=150)
plt.close()
