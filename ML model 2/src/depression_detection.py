import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from utils import clean_text

RANDOM_STATE = 42

def main():
    # Load dataset
    df = pd.read_csv("data/tweets_combined.csv")
    df = df.dropna(subset=["tweet", "target"])
    df["tweet"] = df["tweet"].astype(str)
    df["target"] = df["target"].astype(int)

    X = df["tweet"].apply(clean_text)
    y = df["target"]

    print(f"Dataset size: {len(df)}  |  Positives: {y.sum()}  Negatives: {len(y)-y.sum()}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1,2), 
        max_features=20000, 
        stop_words="english"
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # Logistic Regression with hyperparameter tuning
    param_grid = {
        "C": [0.1, 1, 5],
        "penalty": ["l2"],
        "solver": ["liblinear", "saga"]
    }
    model = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)
    grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_tfidf, y_train)

    best_model = grid.best_estimator_
    print(f"Best parameters: {grid.best_params_}")

    # Evaluate
    y_pred = best_model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save
    joblib.dump(best_model, "models/depression_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("\nModel and vectorizer saved in 'models/' folder.")

if __name__ == "__main__":
    main()
