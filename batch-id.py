# src/batch_predict_interactive.py

import fasttext
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import os

def safe_predict(model, text, k=1):
    """
    Safe prediction function to handle NumPy/FastText quirks
    """
    try:
        labels, probs = model.predict(text, k=k)
        return [(label.replace('__label__', ''), prob) for label, prob in zip(labels, probs)]
    except Exception:
        return [("error", 0.0)]

def choose_column(columns):
    print("\n📋 Available columns in the CSV:")
    for i, col in enumerate(columns):
        print(f"  {i + 1}. {col}")
    
    while True:
        try:
            choice = int(input("\n🔢 Enter the number of the column that contains the sentences: "))
            if 1 <= choice <= len(columns):
                return columns[choice - 1]
            else:
                print("❗ Invalid selection. Try again.")
        except ValueError:
            print("❗ Please enter a number.")

def main():
    print("🟡 Twi Language Batch Detector")

    # Initialize Tkinter root and hide the main window
    root = tk.Tk()
    root.withdraw()

    # Prompt for CSV input file using file dialog
    print("\n📥 Please select the input CSV file...")
    input_path = filedialog.askopenfilename(
        title="Select Input CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if not input_path:
        print("❌ No file selected. Exiting.")
        return

    input_file = Path(input_path)

    # Load CSV
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return

    # Ask user to pick column
    text_column = choose_column(df.columns)

    # Create data folder if it doesn't exist
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)

    # Generate output filename based on input filename
    input_filename = input_file.stem
    output_filename = f"{input_filename}_predictions.csv"
    output_file = data_folder / output_filename

    # Check if output file exists and prompt for overwrite
    if output_file.exists():
        overwrite = input(f"\n⚠️ File {output_file} already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("❌ Operation cancelled.")
            return

    # Load model
    model_path = Path("model/twi_id_model.bin")
    if not model_path.exists():
        print("❌ Model not found at 'model/twi_id_model.bin'.")
        return

    print("\n✅ Loading model...")
    model = fasttext.load_model(str(model_path))

    print(f"🔎 Running predictions on {len(df)} rows from column '{text_column}'...")

    predictions = df[text_column].astype(str).apply(lambda x: safe_predict(model, x, k=1)[0])
    df['predicted_lang'] = predictions.apply(lambda x: x[0])
    df['confidence'] = predictions.apply(lambda x: x[1])

    df.to_csv(output_file, index=False)
    print(f"\n✅ Done! Predictions saved to: {output_file}")

if __name__ == "__main__":
    main()
