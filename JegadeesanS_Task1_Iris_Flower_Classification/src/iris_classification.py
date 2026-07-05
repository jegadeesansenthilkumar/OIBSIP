# Iris Flower Classification - Task 1
# Author: Jegadeesan S

import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Create required folders
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("IRIS FLOWER CLASSIFICATION")
print("=" * 60)

# Load Iris dataset
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species_name"] = df["species"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nNull Values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe())

# Save dataset overview
df.to_csv("iris_dataset.csv", index=False)

# Pairplot
sns.pairplot(df, hue="species_name", vars=iris.feature_names)
plt.savefig("images/pairplot.png")
plt.close()

# Boxplots
for feature in iris.feature_names:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="species_name", y=feature, data=df)
    plt.title(f"Boxplot of {feature} by Species")
    plt.savefig(f"images/boxplot_{feature.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[iris.feature_names].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.close()

# Features and target
X = df[iris.feature_names]
y = df["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}

print("\nModel Evaluation Results")
print("-" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(f"\n{name}")
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"images/confusion_matrix_{name.replace(' ', '_').replace('-', '')}.png")
    plt.close()

# Accuracy comparison
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
results_df.to_csv("model_results.csv", index=False)

plt.figure(figsize=(9, 5))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=20)
plt.ylim(0, 1.1)
plt.savefig("images/model_accuracy_comparison.png")
plt.close()

# Select best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\n" + "=" * 60)
print("Best Model:", best_model_name)
print("Best Accuracy:", results[best_model_name])
print("=" * 60)

# Save best model
joblib.dump(best_model, "models/iris_classifier.pkl")

print("\nBest model saved successfully in models/iris_classifier.pkl")
print("All visualizations saved in images folder.")
print("Project completed successfully.")
