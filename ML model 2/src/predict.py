import sys
import joblib
from utils import clean_text

def load_model():
    model = joblib.load("models/depression_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

def predict_text(text: str):
    model, vectorizer = load_model()
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    label = "DEPRESSED (1)" if pred == 1 else "NOT depressed (0)"
    return label

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your text here\"")
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    result = predict_text(input_text)
    print(f"\nInput: {input_text}\nPrediction: {result}\n")
