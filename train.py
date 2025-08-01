# Step 2: Import libraries
import pandas as pd
import fasttext
from sklearn.model_selection import train_test_split
import numpy as np

# Step 3: Specify your CSV path
csv_path = "sentences.csv"  # CHANGE THIS TO YOUR CSV PATH

# Step 4: Read and prepare dataset
df = pd.read_csv(csv_path)

# Convert to FastText format
df['fasttext_format'] = '__label__' + df['lang'] + ' ' + df['sentence'].astype(str).str.replace('\n', ' ')

# Step 5: Split data
train, test = train_test_split(df['fasttext_format'], test_size=0.2, random_state=42)

# Save datasets
train_file = "fasttext_train.txt"
test_file = "fasttext_test.txt"

train.to_csv(train_file, index=False, header=False, encoding='utf-8')
test.to_csv(test_file, index=False, header=False, encoding='utf-8')

print(f"\nDataset sizes:\nTrain: {len(train):,} samples\nTest: {len(test):,} samples")

# Step 6: Train FastText model
model = fasttext.train_supervised(
    input=train_file,
    lr=0.5,
    epoch=25,
    wordNgrams=2,
    dim=100,
    loss='ova',
    thread=8
)

# Step 7: Evaluate the model
results = model.test(test_file)
print(f"\nEvaluation results:\nPrecision: {results[1]:.4f}\nRecall: {results[2]:.4f}\nNumber of examples: {results[0]:,}")

# Step 8: Save the model
model.save_model("twi_id_model.bin")
print("\nModel saved'")

# Step 9: Fixed prediction example
test_sentence = "Gâtâ be gyu dûkô fô bâ ma sara ditele"

# Solution 1: Use try-except block to handle the error gracefully
try:
    labels, probabilities = model.predict(test_sentence, k=1)
    print(f"\nPrediction for test sentence: {test_sentence}")
    print(f"Predicted language: {labels[0]}")
    print(f"Probability: {probabilities[0]:.4f}")
except ValueError as e:
    print(f"Prediction error: {e}")
    # Alternative approach using model.predict with different parameters
    try:
        prediction = model.predict(test_sentence)
        print(f"\nPrediction for test sentence: {test_sentence}")
        print(f"Raw prediction: {prediction}")
    except Exception as e2:
        print(f"Alternative prediction also failed: {e2}")

# Step 10: Additional test with multiple predictions
print("\n" + "="*50)
print("Testing multiple sentences:")

test_sentences = [
    "Hello, how are you today?",
    "Bonjour, comment allez-vous?",
    "Hola, ¿cómo estás?",
    "Guten Tag, wie geht es Ihnen?",
    "Ciao, come stai?"
]

for sentence in test_sentences:
    try:
        labels, probabilities = model.predict(sentence, k=3)  # Get top 3 predictions
        print(f"\nSentence: {sentence}")
        print("Top predictions:")
        for i, (label, prob) in enumerate(zip(labels, probabilities)):
            clean_label = label.replace('__label__', '')
            print(f"  {i+1}. {clean_label}: {prob:.4f}")
    except ValueError:
        # Fallback method
        try:
            prediction = model.predict(sentence)
            labels = prediction[0] if isinstance(prediction, tuple) else [prediction]
            print(f"\nSentence: {sentence}")
            print(f"Predicted language: {labels[0].replace('__label__', '') if labels else 'Unknown'}")
        except Exception as e:
            print(f"\nSentence: {sentence}")
            print(f"Prediction failed: {e}")

# Step 11: Function for safe prediction
def safe_predict(model, text, k=1):
    """
    Safe prediction function that handles NumPy compatibility issues
    """
    try:
        labels, probabilities = model.predict(text, k=k)
        return [(label.replace('__label__', ''), prob) for label, prob in zip(labels, probabilities)]
    except ValueError:
        # Fallback for NumPy 2.0 compatibility issue
        try:
            prediction = model.predict(text)
            if isinstance(prediction, tuple) and len(prediction) >= 2:
                labels = prediction[0]
                # Try to extract probabilities if available
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

# Test the safe prediction function
print("\n" + "="*50)
print("Using safe prediction function:")
result = safe_predict(model, test_sentence, k=3)
print(f"\nSentence: {test_sentence}")
for i, (lang, prob) in enumerate(result):
    print(f"  {i+1}. {lang}: {prob:.4f}")
