import pandas as pd

# Load your CSV
df = pd.read_csv('predicted.csv')  # Replace with your actual file name

# Split into two DataFrames
twi_df = df[df['predicted_lang'].str.lower() == 'twi']
non_twi_df = df[df['predicted_lang'].str.lower() != 'twi']

# Save to new CSV files
twi_df.to_csv('twi_sentences.csv', index=False)
non_twi_df.to_csv('non_twi_sentences.csv', index=False)

