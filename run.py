import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
train_path = os.path.join(BASE_DIR, "train_model.py")
app_path = os.path.join(BASE_DIR, "app.py")

# Train only if model doesn't exist
if not os.path.exists(model_path):
    print("Training model...")
    subprocess.run([sys.executable, train_path])
else:
    print("Model already trained. Skipping training.")

# Run app
subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])