#!/usr/bin/env python3
"""
gemini_flash_confusion_matrix.py
--------------------------------
Creates a confusion matrix for the Kaggle Suicide-Watch dataset using
Google's Gemini 2.0 Flash model on Vertex AI.

Prerequisites
-------------
# ➊ Google Cloud project + service account with Vertex AI User role
   gcloud auth application-default login   # or use a service-account key

# ➋ Environment variables (bash):
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"   # or another supported region
export GOOGLE_GENAI_USE_VERTEXAI="true"      # routes google-generativeai to Vertex AI
export KAGGLE_USERNAME="<kaggle-username>"
export KAGGLE_KEY="<kaggle-api-token>"

# ➌ pip install:
python -m pip install --upgrade google-generativeai google-cloud-aiplatform kaggle pandas scikit-learn matplotlib tqdm
"""

import os, tempfile, zipfile, subprocess, pathlib
import pandas as pd
from google import genai
from google.genai.types import HttpOptions, Part
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
from tenacity import retry, stop_after_attempt, wait_exponential

from dotenv import load_dotenv
load_dotenv()
project = os.environ.get("GOOGLE_CLOUD_PROJECT")
location = os.environ.get("GOOGLE_CLOUD_LOCATION")
use_vertexai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"  # Better boolean conversion
print(f"Loaded environment variables:")
print(f"Project: {project}")
print(f"Location: {location}")
print(f"Use VertexAI: {use_vertexai}")

GEMINI_MODEL_VER = 'gemini-2.0-flash-001'
# model = genai.Client(vertexai = True, project = "cs-152-project-460104", location="us-central1").models

# ------------------------------------------------------------------
# 1. Download the data set from Kaggle
# ------------------------------------------------------------------
def download_kaggle_dataset():
    kaggle_ref = "nikhileswarkomati/suicide-watch"
    dst = pathlib.Path(tempfile.gettempdir()) / "suicide_watch"
    dst.mkdir(exist_ok=True)
    zip_path = dst / "suicide-watch.zip"

    if not zip_path.exists():
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_ref, "-p", str(dst), "--force"],
            check=True
        )

    # unzip (file names vary slightly between versions – extract all)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)
    return dst

# ------------------------------------------------------------------
# 2. Load a CSV that contains text & label columns
# ------------------------------------------------------------------
def load_dataset(folder: pathlib.Path):
    # You may need to adjust the file name / columns if the author updates the set
    candidates = list(folder.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError("No CSV file found in the extracted dataset folder")
    df = pd.read_csv(candidates[0])

    # Column names in this dataset: 'text' and 'class' (1 = suicidal)
    text_col = next(c for c in df.columns if c.lower().startswith("text"))
    label_col = next(c for c in df.columns if c.lower().startswith("class"))  # Changed from 'label' to 'class'
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    df["label"] = df["label"].map({"suicide": 1, "non-suicide": 0})
    return df

# ------------------------------------------------------------------
# 3. Initialize Gemini 2.0 Flash
# ------------------------------------------------------------------
def init_gemini():
    # Requires GOOGLE_CLOUD_PROJECT & friends in env
    model = genai.Client(vertexai = use_vertexai, project = project, location=location).models
    return model

# ------------------------------------------------------------------
# 4. Ask Gemini to classify each post
# ------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def classify(model, prompt_text):
    """Returns 'positive' or 'negative'."""
    # system_prompt = f"""
    #     You are a mental-health content classifier.
    #     Respond ONLY with the single word 'positive' if the text implies self-harm
    #     or suicidal ideation, otherwise respond 'negative'. No other words.

    #     Text: {prompt_text}
    # """
    prompt = f"""
        Analyze the following message and determine if it contains content related to suicide or self-harm.
        Return only "YES" if the message contains suicide or self-harm related content, or "NO" if it does not.

        Message: {prompt_text}
        """
    
    try:
        response = model.generate_content(
            model=GEMINI_MODEL_VER,
            contents=prompt
        )
        answer = response.text.strip().lower()
        return "positive" if "yes" in answer else "negative"
    except Exception as e:
        print(f"Error classifying text: {str(e)}")
        raise

# ------------------------------------------------------------------
# 5. Main orchestration
# ------------------------------------------------------------------
def main(sample_size=None):
    folder = download_kaggle_dataset()
    df = load_dataset(folder)

    # Optional: subsample for cost control
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    model = init_gemini()
    
    preds = []
    total_samples = len(df)
    print(f"\nProcessing {total_samples} samples for classification...")
    
    for idx, text in enumerate(df["text"]):
        print(f"\rProcessing sample {idx+1}/{total_samples} ({(idx+1)/total_samples*100:.1f}%)...", end="", flush=True)
        try:
            result = classify(model, text)
            preds.append(result)
        except Exception as e:
            print(f"\nFailed to process item {idx}: {str(e)}")
            preds.append("negative")  # Default to negative on error
    print("\nAll samples processed.")

    df["pred"] = (pd.Series(preds) == "positive").astype(int)

    # ------------------------------------------------------------------
    # Print false positives and false negatives
    # ------------------------------------------------------------------
    print("\n--- FALSE POSITIVES (Predicted positive but actually negative) ---")
    false_positives = df[(df["label"] == 0) & (df["pred"] == 1)]
    
    # Open text file for writing results
    with open("classification_errors.txt", "w") as error_file:
        error_file.write("--- FALSE POSITIVES (Predicted positive but actually negative) ---\n\n")
        
        for i, row in false_positives.iterrows():
            # Print to console
            print(f"Index: {i}")
            print("FALSE POSTIIVE: ")
            print(f"Text: {row['text']}")
            print("-" * 50)
            
            # Write to file
            error_file.write(f"Index: {i}\n")
            error_file.write("FALSE POSITIVE:\n")
            error_file.write(f"Text: {row['text']}\n")
            error_file.write("-" * 50 + "\n\n")

        error_file.write("\n\n--- FALSE NEGATIVES (Predicted negative but actually positive) ---\n\n")
        
        print("\n--- FALSE NEGATIVES (Predicted negative but actually positive) ---")
        false_negatives = df[(df["label"] == 1) & (df["pred"] == 0)]
        
        for i, row in false_negatives.iterrows():
            # Print to console
            print(f"Index: {i}")
            print("FALSE NEGATIVE: ")
            print(f"Text: {row['text']}")
            print("-" * 50)
            
            # Write to file
            error_file.write(f"Index: {i}\n")
            error_file.write("FALSE NEGATIVE:\n")
            error_file.write(f"Text: {row['text']}\n")
            error_file.write("-" * 50 + "\n\n")
    
    print(f"\nClassification errors have been saved to 'classification_errors.txt'")

    # ------------------------------------------------------------------
    # 6. Confusion matrix
    # ------------------------------------------------------------------
    cm = confusion_matrix(df["label"], df["pred"], labels=[1, 0])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Positive", "Negative"])
    disp.plot(values_format="d")
    plt.title("Gemini 2.0 Flash confusion matrix on Suicide-Watch")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set sample_size=None to evaluate on the full set (can be thousands of calls!)
    main(sample_size=500)
