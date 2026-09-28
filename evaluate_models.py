import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD MODEL RESULTS
# ==========================================

results = pd.read_csv("model_results.csv")

print("Model results loaded successfully!")

print("\n========== MODEL RESULTS ==========")
print(results.round(4))


# ==========================================
# 2. MODEL COMPARISON GRAPH
# ==========================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

results_plot = results.set_index("Model")[metrics]

results_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Rainfall Prediction Model Comparison")

plt.xlabel("Machine Learning Model")

plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(rotation=20)

plt.legend(
    title="Metrics"
)

plt.tight_layout()

plt.savefig(
    "model_comparison.png",
    dpi=300
)

plt.close()

print("\nModel comparison graph saved!")


# ==========================================
# 3. F1 SCORE COMPARISON
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    results["Model"],
    results["F1 Score"]
)

plt.title("F1 Score Comparison")

plt.xlabel("Model")

plt.ylabel("F1 Score")

plt.ylim(0, 1)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "f1_score_comparison.png",
    dpi=300
)

plt.close()

print("F1 score graph saved!")


# ==========================================
# 4. BEST MODEL
# ==========================================

best_row = results.loc[
    results["F1 Score"].idxmax()
]

print("\n===================================")
print("BEST MODEL")
print("===================================")

print(
    "Model:",
    best_row["Model"]
)

print(
    "Accuracy:",
    round(best_row["Accuracy"], 4)
)

print(
    "Precision:",
    round(best_row["Precision"], 4)
)

print(
    "Recall:",
    round(best_row["Recall"], 4)
)

print(
    "F1 Score:",
    round(best_row["F1 Score"], 4)
)


# ==========================================
# 5. COMPLETED
# ==========================================

print("\n===================================")
print("MODEL EVALUATION COMPLETED!")
print("===================================")

print("\nCreated files:")

print("1. model_comparison.png")

print("2. f1_score_comparison.png")