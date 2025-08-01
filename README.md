# Twi Language Identification Model 🇬🇭

This repository contains a simple and fast **language identification system** for detecting whether a sentence is in **Twi (Akan)** or not. It uses **FastText** for both training and inference and is designed to be easily adapted for other low-resource languages.

---

## 🧠 What It Does

- Trains a binary classifier to detect Twi language sentences.
- Includes a pre-trained model for immediate use.
- Supports **batch** and **single sentence** prediction.
- Accepts CSV uploads and allows users to **predict in bulk**.
- Easily adaptable to other languages with your own training data.

---

## 🗂️ Project Structure

```
twi-id/
├── data/
│   ├── train.csv                # Your training data (Twi/Non-Twi labeled)
│   └── sample_sentences.csv     # Sample file for testing predictions
├── model/
│   └── twi_id_model.bin         # Trained FastText model
├── id_single-sentence.py        # Predicts one sentence at a time
├── id_multiple-sentences.py     # Prompts user to upload a CSV and outputs predictions
├── train-model.py               # Script to train the model
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/GhanaNLP/twi-lang-id.git
cd twi-lang-id
```

2. **(Recommended) Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

> If you run into issues, install FastText separately:
> ```bash
> pip install fasttext
> ```

---

## 🚀 Usage

### 🔹 Predict a Single Sentence

```bash
python3 id_single-sentence.py
```

You'll be prompted to enter a sentence and the script will return whether it's in Twi or not.

---

### 🔹 Predict from CSV (Batch)

```bash
python3 id_multiple-sentences.py
```

This script will:
1. Prompt you to select a CSV file (e.g. `sample_sentences.csv`)
2. Ask you to choose the column that contains the text
3. Output predictions to a new CSV you can name and save

---

### 🔹 Train Your Own Model

To train on new data (e.g., for another language):

1. Prepare a `train.csv` with the following format:
    ```
    __label__twi    sentence in Twi
    __label__not    sentence in English
    ```
2. Run the training script:
    ```bash
    python3 train-model.py
    ```

This will output a `twi_id_model.bin` file inside the `model/` directory.

---

## 🧪 Sample Test File

You can test batch predictions with the sample provided:

```bash
data/sample_sentences.csv
```

---

## ⚙️ Regenerating Requirements

If you add new dependencies, regenerate `requirements.txt` with:

```bash
pipreqs . --force
```

---

## 🔒 License

MIT License — feel free to fork, adapt, and build upon it!

---

## 🙌 Acknowledgements

- Built with [FastText](https://fasttext.cc/)
- Part of the [GhanaNLP](https://github.com/GhanaNLP) open language tech ecosystem

