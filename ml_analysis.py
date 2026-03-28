# ==========================================
# FINAL ML ANALYSIS CODE (NON-BLOCKING)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# IMPORTANT: prevents graph blocking
plt.ion()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

# ==========================================
# 1. LOAD DATASET
# ==========================================
data = pd.read_csv("health_data.csv")

print("\nDataset Preview:\n", data.head())
print("\nColumns:\n", data.columns)

# ❗ Remove ID if exists
if "ID" in data.columns:
    data = data.drop("ID", axis=1)

# ==========================================
# 2. DEFINE FEATURES & TARGET
# ==========================================
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

feature_names = X.columns

# ==========================================
# 3. DATA SPLIT
# ==========================================
split_ratio = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split_ratio, random_state=42
)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 4. MODELS
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "SVM": SVC(probability=True)
}

results = {}

# ==========================================
# 5. TRAIN + METRICS + K-FOLD
# ==========================================
for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # ✅ Correct K-Fold (use training data only)
    kfold = cross_val_score(model, X_train, y_train, cv=5)
    kfold_mean = kfold.mean()

    results[name] = [acc, prec, rec, f1, kfold_mean]

    print("\n======", name, "======")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("K-Fold Accuracy:", kfold_mean)

# ==========================================
# 6. BEST MODEL
# ==========================================
best_model_name = max(results, key=lambda x: results[x][0])
best_model = models[best_model_name]

print("\n Best Model:", best_model_name)

# ==========================================
# 7. CONFUSION MATRIX
# ==========================================
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix - " + best_model_name)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.savefig("confusion_matrix.png")

# ==========================================
# 8. ROC CURVE
# ==========================================
y_prob = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve (AUC = %.2f)" % roc_auc)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("roc_curve.png")

# ==========================================
# 9. MODEL COMPARISON GRAPH
# ==========================================
labels = list(results.keys())

accuracy = [results[m][0] for m in labels]
precision = [results[m][1] for m in labels]
recall = [results[m][2] for m in labels]
f1 = [results[m][3] for m in labels]

x = np.arange(len(labels))

plt.figure()
plt.bar(x, accuracy, width=0.2)
plt.bar(x + 0.2, precision, width=0.2)
plt.bar(x + 0.4, recall, width=0.2)
plt.bar(x + 0.6, f1, width=0.2)

plt.xticks(x, labels)
plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.legend(["Accuracy", "Precision", "Recall", "F1"])
plt.savefig("model_comparison.png")

# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_

plt.figure()
plt.barh(feature_names, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.savefig("feature_importance.png")

# ==========================================
# 11. SAVE RESULTS
# ==========================================
df = pd.DataFrame(
    results,
    index=["Accuracy", "Precision", "Recall", "F1", "K-Fold"]
)

df.to_csv("results.csv")

print("\nAll graphs saved in project folder!")
print("Files generated:")
print("confusion_matrix.png")
print("roc_curve.png")
print("model_comparison.png")
print("feature_importance.png")

# ==========================================
# SHOW ALL GRAPHS
# ==========================================
plt.show()