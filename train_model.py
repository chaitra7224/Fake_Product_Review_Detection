import os
import re
import json
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. FOLDER SETTINGS
DATASET_PATH = os.path.join("dataset", "review_dataset.csv")
MODEL_FOLDER = "models"

os.makedirs(MODEL_FOLDER, exist_ok=True)

# 2. TEXT CLEANING FUNCTION

def clean_text(text):
    """
    Cleans review text before machine learning.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Keep letters and numbers
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# 3. LOAD DATASET

print("\n" + "=" * 60)
print("FAKE PRODUCT REVIEW DETECTION")
print("=" * 60)

print("\nLoading dataset...")

if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset not found!")
    print("Please put your Kaggle CSV here:")
    print("dataset/reviews.csv")
    exit()

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# 4. CHECK REQUIRED COLUMNS

required_columns = ["rating", "label", "text_"]

for column in required_columns:
    if column not in df.columns:
        print(f"\nERROR: Required column '{column}' is missing.")
        print("Available columns:", df.columns.tolist())
        exit()

# 5. SELECT REQUIRED DATA

df = df[["rating", "label", "text_"]].copy()

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 6. REMOVE MISSING VALUES

df.dropna(subset=["label", "text_"], inplace=True)

# Remove empty reviews
df["text_"] = df["text_"].astype(str)

df = df[df["text_"].str.strip() != ""]

df.reset_index(drop=True, inplace=True)

# 7. CLEAN REVIEW TEXT

print("\nCleaning review text...")

df["clean_text"] = df["text_"].apply(clean_text)

# 8. CONVERT LABELS

print("\nOriginal labels:")
print(df["label"].value_counts())

# CG = Computer Generated / Fake
# OR = Original / Genuine

label_mapping = {
    "CG": 1,
    "OR": 0
}

df["target"] = df["label"].str.upper().map(label_mapping)

# Remove unknown labels
df = df.dropna(subset=["target"])

df["target"] = df["target"].astype(int)

print("\nConverted labels:")
print(df["target"].value_counts())

print("\nLabel meaning:")
print("0 = Original Review")
print("1 = Fake / Computer Generated Review")

# 9. TRAIN / TEST SPLIT

X = df["clean_text"]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining reviews:", len(X_train))
print("Testing reviews :", len(X_test))

# 10. TF-IDF FEATURE EXTRACTION

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape :", X_test_tfidf.shape)


# Save vectorizer
joblib.dump(
    vectorizer,
    os.path.join(MODEL_FOLDER, "tfidf_vectorizer.pkl")
)

# 11. DECISION TREE MODEL

print("\n" + "-" * 60)
print("Training Decision Tree Classifier...")
print("-" * 60)

start_time = time.time()

decision_tree = DecisionTreeClassifier(
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

decision_tree.fit(X_train_tfidf, y_train)

dt_training_time = time.time() - start_time

dt_prediction = decision_tree.predict(X_test_tfidf)

# 12. DECISION TREE METRICS

dt_accuracy = accuracy_score(y_test, dt_prediction)

dt_precision = precision_score(
    y_test,
    dt_prediction,
    zero_division=0
)

dt_recall = recall_score(
    y_test,
    dt_prediction,
    zero_division=0
)

dt_f1 = f1_score(
    y_test,
    dt_prediction,
    zero_division=0
)

dt_confusion = confusion_matrix(
    y_test,
    dt_prediction
)

print("\nDecision Tree Results")
print("-" * 40)
print(f"Accuracy  : {dt_accuracy:.4f}")
print(f"Precision : {dt_precision:.4f}")
print(f"Recall    : {dt_recall:.4f}")
print(f"F1 Score  : {dt_f1:.4f}")
print(f"Training Time: {dt_training_time:.2f} seconds")

print("\nDecision Tree Confusion Matrix:")
print(dt_confusion)


# Save Decision Tree
joblib.dump(
    decision_tree,
    os.path.join(MODEL_FOLDER, "decision_tree_model.pkl")
)

# 13. ADABOOST MODEL

print("\n" + "-" * 60)
print("Training AdaBoost Classifier...")
print("-" * 60)

start_time = time.time()

base_tree = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

adaboost = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=50,
    learning_rate=0.8,
    random_state=42
)

adaboost.fit(X_train_tfidf, y_train)

ada_training_time = time.time() - start_time

ada_prediction = adaboost.predict(X_test_tfidf)

# 14. ADABOOST METRICS

ada_accuracy = accuracy_score(y_test, ada_prediction)

ada_precision = precision_score(
    y_test,
    ada_prediction,
    zero_division=0
)

ada_recall = recall_score(
    y_test,
    ada_prediction,
    zero_division=0
)

ada_f1 = f1_score(
    y_test,
    ada_prediction,
    zero_division=0
)

ada_confusion = confusion_matrix(
    y_test,
    ada_prediction
)

print("\nAdaBoost Results")
print("-" * 40)
print(f"Accuracy  : {ada_accuracy:.4f}")
print(f"Precision : {ada_precision:.4f}")
print(f"Recall    : {ada_recall:.4f}")
print(f"F1 Score  : {ada_f1:.4f}")
print(f"Training Time: {ada_training_time:.2f} seconds")

print("\nAdaBoost Confusion Matrix:")
print(ada_confusion)


# Save AdaBoost
joblib.dump(
    adaboost,
    os.path.join(MODEL_FOLDER, "adaboost_model.pkl")
)

# 15. CLASSIFICATION REPORTS

print("\n" + "=" * 60)
print("DECISION TREE CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        dt_prediction,
        target_names=["Original", "Fake"],
        zero_division=0
    )
)


print("\n" + "=" * 60)
print("ADABOOST CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        ada_prediction,
        target_names=["Original", "Fake"],
        zero_division=0
    )
)

# 16. SAVE MODEL PERFORMANCE

results = {
    "dataset": {
        "total_reviews": int(len(df)),
        "training_reviews": int(len(X_train)),
        "testing_reviews": int(len(X_test)),
        "fake_reviews": int((df["target"] == 1).sum()),
        "original_reviews": int((df["target"] == 0).sum())
    },

    "decision_tree": {
        "accuracy": round(float(dt_accuracy), 4),
        "precision": round(float(dt_precision), 4),
        "recall": round(float(dt_recall), 4),
        "f1_score": round(float(dt_f1), 4),
        "training_time": round(float(dt_training_time), 2),
        "confusion_matrix": dt_confusion.tolist()
    },

    "adaboost": {
        "accuracy": round(float(ada_accuracy), 4),
        "precision": round(float(ada_precision), 4),
        "recall": round(float(ada_recall), 4),
        "f1_score": round(float(ada_f1), 4),
        "training_time": round(float(ada_training_time), 2),
        "confusion_matrix": ada_confusion.tolist()
    }
}

with open(
    os.path.join(MODEL_FOLDER, "performance.json"),
    "w"
) as file:
    json.dump(results, file, indent=4)

# 17. SAVE TEST DATA FOR ANALYTICS

test_results = pd.DataFrame({
    "review": X_test.values,
    "actual": y_test.values,
    "decision_tree_prediction": dt_prediction,
    "adaboost_prediction": ada_prediction
})

test_results.to_csv(
    os.path.join(MODEL_FOLDER, "test_predictions.csv"),
    index=False
)

# 18. FINAL OUTPUT

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nSaved files:")

print("1. models/tfidf_vectorizer.pkl")
print("2. models/decision_tree_model.pkl")
print("3. models/adaboost_model.pkl")
print("4. models/performance.json")
print("5. models/test_predictions.csv")

print("\nBest model:")

if ada_accuracy >= dt_accuracy:
    print("AdaBoost Classifier")
    print(f"Accuracy: {ada_accuracy:.2%}")
else:
    print("Decision Tree Classifier")
    print(f"Accuracy: {dt_accuracy:.2%}")

print("\nYou can now proceed to the Flask web application.")