#!/usr/bin/env python3
"""
Evaluate Gemini on the validation split and build a 6×6 confusion matrix.

Author: <your-name>
Date  : 2025-05-27
"""

import os
import re
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from google import genai
from google.genai.types import HttpOptions, Part

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


# ────────────────────────────────────────────────────────────────────────────────
# Configuration –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
DATA_DIR          = Path("./")          # folder containing .xlsx and .pkl
EXCEL_FILE        = DATA_DIR / "only_clean_bot_data.xlsx"
CHECKPOINT_DIR    = DATA_DIR                       # where data_split.pkl lives
OUTPUT_DIR        = Path("./gemini_eval_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_TOKENS        = 4              # we only want a single digit back
# ────────────────────────────────────────────────────────────────────────────────


def load_validation_subset() -> pd.DataFrame:
    """Load the Excel sheet and keep only the validation messages with valid labels."""
    # 1. Excel with full dataset
    df = pd.read_excel(EXCEL_FILE, engine="openpyxl")

    # 2. Validation message list from the pickle
    pkl_path = Path(CHECKPOINT_DIR) / "bot_data_split.pkl"
    with open(pkl_path, "rb") as f:
        data_split = pickle.load(f)
    val_messages = set(data_split["val_messages"])
    print("val_messages: ", len(val_messages))
    # 3. Sub-select
    val_df = df[df["message"].isin(val_messages)].copy()

    return val_df


def init_vertex():
    # Requires GOOGLE_CLOUD_PROJECT & friends in env
    model = genai.Client(vertexai = use_vertexai, project = project, location=location).models
    return model

def classify_message(model, text: str) -> int:
    # Ask Gemini to return ONLY "YES" or "NO"
    # Falls back to -1 on parsing problems.

    prompt = f"""
    Analyze the following message and determine if it contains any content related to suicide or self-harm, which can include loneliness and depression.
    Return only "YES" if the message contains any mention of suicide or self-harm related content, or "NO" if it does not.
    
    Message: {text}
    """

    response = model.generate_content(
        model=GEMINI_MODEL_VER,
        contents=prompt
    )
    
    result = response.text.strip().upper()
    print("gemini response: ", result)
    return result == "YES"

def main() -> None:
    df = load_validation_subset()
    print(f"Validation set size (after dropping -1): {len(df):,}")

    model = init_vertex()

    preds = []
    for msg in df["message"]:
        label = classify_message(model, msg)
        preds.append(label)

    df["predicted_category"] = preds

    # ── Confusion matrix ────────────────────────────────────────────────────────
    y_true = df["training_category"].astype(int)
    y_pred = df["predicted_category"].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(2)))
    cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in range(2)],
                             columns=[f"pred_{i}" for i in range(2)])
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv", index=True)
    print("\nConfusion matrix (saved to confusion_matrix.csv):\n")
    print(cm_df)

    # Optional: detailed metrics
    report = classification_report(
        y_true, y_pred, labels=list(range(2)), digits=3, output_dict=False
    )
    print("\nClassification report:\n")
    print(report)

    # ── Save mis-classified messages ───────────────────────────────────────────
    mismatches = df[df["predicted_category"] != df["training_category"]]
    mismatches.to_excel(OUTPUT_DIR / "mismatches.xlsx", index=False)
    print(f"\nSaved {len(mismatches):,} mis-classified messages to mismatches.xlsx")


if __name__ == "__main__":
    main()
