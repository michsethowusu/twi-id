# src/predict.py

import fasttext
import sys
from pathlib import Path

def safe_predict(model, text, k=1):
    """
    Safe prediction function that handles NumPy compatibility issues
    """
    try:
        labels, probabilities = model.predict(text, k=k)
        return [(label.replace('__label__', ''), prob) for label, prob in zip(labels, probabilities)]
    except ValueError:
        try:
            prediction = model.predict(text)
            if isinstance(prediction, tuple) and len(prediction) >= 2:
                labels = prediction[0]
                try:
                    probabilities = prediction[1]
                    return [(label.replace('__label__', ''), prob) for label, prob in zip(labels[:k], probabilities[:k])]
                except:
                    return [(label.replace('__label__', ''), 0.0) for label in labels[:k]]
            else:
                return [("unknown", 0.0)]
        except Exception as e:
            print(f"Prediction error: {e}")
            return [("error", 0.0)]

def main():
    # Set model path
    model_path = Path("model/twi_id_model.bin")

    if not model_path.exists():
        print("❌ Model file not found! Please ensure 'model/twi_id_model.bin' exists.")
        return

    # Load model
    model = fasttext.load_model(str(model_path))

    # Get input text
    if len(sys.argv) > 1:
        # Use command-line argument if provided
        input_text = " ".join(sys.argv[1:])
    else:
        # Prompt user if no argument given
        input_text = input("\nEnter a sentence to analyze: ").strip()
        if not input_text:
            print("⚠️  No input provided. Exiting.")
            return

    predictions = safe_predict(model, input_text, k=3)

    print(f"\n🔎 Input: {input_text}\nTop Predictions:")
    for i, (label, prob) in enumerate(predictions):
        print(f"  {i+1}. {label} (confidence: {prob:.4f})")

if __name__ == "__main__":
    main()
