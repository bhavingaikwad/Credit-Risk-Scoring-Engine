import pandas as pd
import numpy as np
import sqlite3
import os

# --- 1. SETUP ---
# Ensure the 'data' folder exists
if not os.path.exists('data'):
    os.makedirs('data')

print("⚙️ Generating synthetic banking data...")

# --- 2. GENERATE SYNTHETIC DATA ---
# We create 1,000 sample customers to mimic the German Credit Dataset
np.random.seed(42)
n_samples = 1000

data = {
    'Age': np.random.randint(19, 75, n_samples),
    'Credit amount': np.random.randint(500, 15000, n_samples),
    'Duration': np.random.randint(6, 60, n_samples),
    'Sex': np.random.choice(['male', 'female'], n_samples),
    'Purpose': np.random.choice(['car', 'radio/TV', 'furniture', 'business'], n_samples),
    'Risk': np.random.choice(['good', 'bad'], n_samples, p=[0.7, 0.3]) # 70% Good, 30% Bad
}

df = pd.DataFrame(data)

# Save as CSV just for reference (optional)
df.to_csv('data/german_credit_data.csv', index=False)
print("✅ CSV file saved to /data folder.")

# --- 3. INGEST INTO SQL (The Engineering Part) ---
# Connect to SQLite (Creates the file if missing)
db_path = 'data/credit_risk.db'
conn = sqlite3.connect(db_path)

# Dump data into a table named 'loans'
df.to_sql('loans', conn, if_exists='replace', index=False)

conn.close()
print(f"✅ SUCCESS! Database created at: {db_path}")
print("   (You are now ready to train your model.)")