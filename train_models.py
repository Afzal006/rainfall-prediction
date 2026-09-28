# ==========================================
# RAINFALL PREDICTION
# MACHINE LEARNING MODEL TRAINING
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/weather.csv")

print("Dataset loaded successfully!")


# ==========================================
# 2. SELECT FEATURES
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


df = df[features + [target]].copy()


# ==========================================
# 3. REMOVE MISSING TARGET
# ==========================================

df = df.dropna(subset=[target])


# ==========================================
# 4. CONVERT TARGET
# ==========================================

df[target] = df[target].map({
    "No": 0,
    "Yes": 1
})


# ==========================================
# 5. HANDLE NUMERICAL MISSING VALUES
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
# 6. HANDLE RAIN TODAY
# ==========================================

df["RainToday"] = df["RainToday"].fillna(
    df["RainToday"].mode()[0]
)

df["RainToday"] = df["RainToday"].map({
    "No": 0,
    "Yes": 1
})


# ==========================================
# 7. CREATE X AND Y
# ==========================================

X = df[features]

y = df[target]


# ==========================================
# 8. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 9. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# 10. CREATE MODELS
# ==========================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}


# ==========================================
# 11. TRAIN MODELS
# ==========================================

results = []


for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(
        X_test_scaled
    )


    # ======================================
    # METRICS
    # ======================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))


    # ======================================
    # CONFUSION MATRIX
    # ======================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix:")
    print(cm)


    # ======================================
    # STORE RESULTS
    # ======================================

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1
    })


# ==========================================
# 12. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)


print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(
    results_df.round(4)
)


# ==========================================
# 13. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "model_results.csv",
    index=False
)

print(
    "\nModel results saved as model_results.csv"
)


# ==========================================
# 14. SELECT BEST MODEL USING F1 SCORE
# ==========================================

best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

print(
    "\nBest model based on F1 Score:",
    best_model_name
)


# ==========================================
# 15. RETRAIN BEST MODEL
# ==========================================

best_model = models[best_model_name]

best_model.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# 16. SAVE MODEL
# ==========================================

joblib.dump(
    best_model,
    "rainfall_model.pkl"
)

joblib.dump(
    scaler,
    "rainfall_scaler.pkl"
)


print("\n==========================================")
print("MODEL SAVED SUCCESSFULLY!")
print("==========================================")

print("rainfall_model.pkl")
print("rainfall_scaler.pkl")