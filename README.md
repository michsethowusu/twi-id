# Twi Language Identification Model

This repository contains a simple and fast **language identification system** for detecting whether a sentence is in **Twi (Akan)** or not. It uses **FastText** for both training and inference and is designed to be easily adapted for other low-resource languages.

This repo can help you to:

- Verify if sentences in your dataset are in Twi or not.
- Train a binary classifier for any other language using your dataset.

---

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/michsethowusu/twi-id.git
cd twi-id
```

2. **Install dependencies**

```bash
pip install numpy==1.26.4 pandas==2.3.1 scikit-learn==1.7.1 fasttext==0.9.3
```


---

## 🚀 Usage

### 🔹 Predict a Single Sentence

```bash
python3 single-id.py
```

You'll be prompted to enter a sentence and the script will return whether it's in Twi or not.

---

### 🔹 Predict from CSV (Batch)

```bash
python3 batch-id.py
```

This script will:
1. Prompt you to select a CSV file (e.g. `sample_sentences.csv`)
2. Ask you to choose the column that contains the text
3. Output predictions to a new CSV you can name and save

---

### 🔹 Train Your Own Model

To train on new data (e.g., for another language):

1. Prepare a `training-data.csv` with the columns and labels as in the training data in the data folder.

2. Run the training script:
    ```bash
    python3 train.py
    ```

This will output a `twi_id_model.bin` file inside the `model/` directory.

---

## 🧪 Sample Test File

You can test batch predictions with the sample provided:

```bash
data/sample_sentences.csv
```

---

## 🔒 License

MIT License — feel free to fork, adapt, and build upon it!

---

## 🙌 Acknowledgements

- Built with [FastText](https://fasttext.cc/)
- Part of the [GhanaNLP's](https://github.com/GhanaNLP) effort to make Ghanaian Languages accesible.

