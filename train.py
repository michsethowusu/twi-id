# Step 2: Import libraries
import pandas as pd
import fasttext
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# Initialize Tkinter without showing main window
root = tk.Tk()
root.withdraw()

print("="*60)
print("Twi Language Identification Model Trainer")
print("="*60)

# Step 3: Prompt user to select training data CSV
print("\n📂 Please select your training data CSV file:")
csv_path = filedialog.askopenfilename(
    title="Select Training Data CSV",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if not csv_path:
    print("❌ No file selected. Exiting program.")
    exit()

print(f"✅ Selected file: {csv_path}")

# Step 4: Set up output directories
BASE_DIR = Path.cwd()

# Create required directories
model_dir = BASE_DIR / "model"
data_dir = BASE_DIR / "data"
model_dir.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)

print(f"\n📁 Using base directory: {BASE_DIR}")
print(f"💾 Model will be saved to: {model_dir}/")
print(f"📊 Output files will be saved to: {data_dir}/")

# Step 5: Read and prepare dataset
try:
    df = pd.read_csv(csv_path)
    print(f"\n✅ Successfully loaded dataset with {len(df)} rows")
except Exception as e:
    print(f"❌ Error loading CSV file: {e}")
    exit()

# Convert to FastText format
df['fasttext_format'] = '__label__' + df['lang'] + ' ' + df['sentence'].astype(str).str.replace('\n', ' ')

# Step 6: Split data
train, test = train_test_split(df['fasttext_format'], test_size=0.2, random_state=42)

# Save datasets with relative paths
train_file = data_dir / "fasttext_train.txt"
test_file = data_dir / "fasttext_test.txt"

train.to_csv(train_file, index=False, header=False, encoding='utf-8')
test.to_csv(test_file, index=False, header=False, encoding='utf-8')

print(f"\nDataset sizes:\nTrain: {len(train):,} samples\nTest: {len(test):,} samples")
print(f"Saved training data to: {train_file}")
print(f"Saved test data to: {test_file}")

# Step 7: Train FastText model
print("\n🔄 Training model... (this may take several minutes)")
model = fasttext.train_supervised(
    input=str(train_file),
    lr=0.5,
    epoch=25,
    wordNgrams=2,
    dim=100,
    loss='ova',
    thread=8
)

# Step 8: Evaluate the model
results = model.test(str(test_file))
print(f"\n📊 Evaluation results:")
print(f"Precision: {results[1]:.4f}")
print(f"Recall: {results[2]:.4f}")
print(f"Number of examples: {results[0]:,}")

# Step 9: Save the model
model_path = model_dir / "twi_id_model.bin"
model.save_model(str(model_path))
print(f"\n💾 Model saved to: {model_path}")

# Step 10: Test prediction
test_sentence = "Gâtâ be gyu dûkô fô bâ ma sara ditele"
print(f"\n🧪 Testing prediction with sample sentence: '{test_sentence}'")

# Prediction function
def safe_predict(model, text, k=1):
    try:
        labels, probabilities = model.predict(text, k=k)
        return [(label.replace('__label__', ''), prob) for label, prob in zip(labels, probabilities)]
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return [("error", 0.0)]

# Get and display prediction
prediction = safe_predict(model, test_sentence, k=3)
print("\n🔎 Prediction results:")
for i, (lang, prob) in enumerate(prediction):
    print(f"  {i+1}. {lang}: {prob:.4f}")

print("\n✅ Training complete! You can now use the model for predictions.")
