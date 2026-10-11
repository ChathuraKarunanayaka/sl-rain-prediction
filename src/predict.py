"""Phase 6 - Pickle loader + single-sample inference. Run:  python src/predict.py

Loads the saved soft-voting model from models/model.pkl and predicts the ONE
record that was kept out of both training and testing.
"""
import pickle

import pandas as pd

import config
from features import MODEL_FEATURES, RAW_INPUT_COLUMNS, build_features


def load_model(path=config.MODEL_FILE):
    """Pickle loader: read the trained pipeline (preprocessing + model) from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_raw(model, raw_row: pd.DataFrame):
    """Predict from raw input fields. Features are rebuilt with the same
    build_features() function that was used for training."""
    features = build_features(raw_row)[MODEL_FEATURES]
    predicted_class = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    return predicted_class, probabilities


def main():
    model = load_model()
    sample = pd.read_csv(config.INFERENCE_SAMPLE_FILE)
    raw_row = sample[RAW_INPUT_COLUMNS]          # only what a user would type in

    predicted, prob = predict_raw(model, raw_row)
    true_label = int(sample[config.TARGET].iloc[0])

    print("Model loaded from:", config.MODEL_FILE.name, "->", type(model.named_steps["model"]).__name__)
    print("\nInput row (held-out sample):")
    print(raw_row.T.to_string(header=False))
    print(f"\nPredicted class : {predicted} ({config.CLASS_LABELS[predicted]})")
    print(f"Probabilities   : No rain = {prob[0]:.3f}, Rain = {prob[1]:.3f}")
    print(f"True label      : {true_label} ({config.CLASS_LABELS[true_label]})")
    print("Result          :", "CORRECT" if predicted == true_label else "INCORRECT")


if __name__ == "__main__":
    main()
