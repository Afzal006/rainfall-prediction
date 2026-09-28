# ==========================================
# RAINFALL PREDICTION
# DATA CLEANING + EDA
# ==========================================

import matplotlib

# Use non-GUI backend
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/weather.csv")

print("Dataset loaded successfully!")


# ==========================================
# 2. DATASET INFORMATION
# ==========================================

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())


# ==========================================
# 3. MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ==========================================
# 4. TARGET DISTRIBUTION
# ==========================================

print("\n========== RAIN TOMORROW DISTRIBUTION ==========")

print(
    df["RainTomorrow"].value_counts()
)


# ==========================================
# 5. RAIN TOMORROW GRAPH
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="RainTomorrow"
)

plt.title(
    "Rain Tomorrow Distribution"
)

plt.xlabel(
    "Rain Tomorrow"
)

plt.ylabel(
    "Number of Records"
)

plt.tight_layout()

plt.savefig(
    "rain_tomorrow_distribution.png",
    dpi=300
)

plt.close()

print(
    "\nRainfall distribution graph saved successfully."
)


# ==========================================
# 6. SELECT FEATURES
# ==========================================

features = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustSpeed",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm",
    "RainToday"
]

target = "RainTomorrow"


df = df[
    features + [target]
].copy()


# ==========================================
# 7. REMOVE MISSING TARGET
# ==========================================

df = df.dropna(
    subset=[target]
)

print(
    "\nRows after removing missing RainTomorrow:"
)

print(
    df.shape
)


# ==========================================
# 8. CONVERT TARGET
# ==========================================

df[target] = df[target].map({
    "No": 0,
    "Yes": 1
})


# ==========================================
# 9. NUMERICAL MISSING VALUES
# ==========================================

numeric_columns = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustSpeed",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm"
]


for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


# ==========================================
# 10. RAIN TODAY
# ==========================================

df["RainToday"] = df["RainToday"].fillna(
    df["RainToday"].mode()[0]
)


df["RainToday"] = df["RainToday"].map({
    "No": 0,
    "Yes": 1
})


# ==========================================
# 11. CHECK MISSING VALUES
# ==========================================

print(
    "\n========== MISSING VALUES AFTER CLEANING =========="
)

print(
    df.isnull().sum()
)


# ==========================================
# 12. FINAL DATASET
# ==========================================

print(
    "\n========== FINAL DATASET =========="
)

print(
    df.head()
)

print(
    "\nFinal dataset shape:"
)

print(
    df.shape
)


# ==========================================
# 13. CORRELATION HEATMAP
# ==========================================

plt.figure(
    figsize=(14, 10)
)

sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    annot=False
)

plt.title(
    "Weather Features Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    "correlation_heatmap.png",
    dpi=300
)

plt.close()

print(
    "\nCorrelation heatmap saved successfully."
)


# ==========================================
# 14. COMPLETED
# ==========================================

print("\n==========================================")
print("DATA CLEANING AND EDA COMPLETED!")
print("==========================================")

print("\nCreated files:")
print("1. rain_tomorrow_distribution.png")
print("2. correlation_heatmap.png")