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
EXCEL_FILE        = DATA_DIR / "user_data.xlsx"
CHECKPOINT_DIR    = DATA_DIR                       # where data_split.pkl lives
OUTPUT_DIR        = Path("./gemini_eval_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PROJECT_ID        = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION          = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# GEMINI_MODEL_NAME = "gemini-1.0-pro"
# TEMPERATURE       = 0.0            # deterministic classification
MAX_TOKENS        = 4              # we only want a single digit back
# ────────────────────────────────────────────────────────────────────────────────


def load_validation_subset() -> pd.DataFrame:
    """Load the Excel sheet and keep only the validation messages with valid labels."""
    # 1. Excel with full dataset
    df = pd.read_excel(EXCEL_FILE, engine="openpyxl")

    # 2. Validation message list from the pickle
    pkl_path = Path(CHECKPOINT_DIR) / "data_split.pkl"
    with open(pkl_path, "rb") as f:
        data_split = pickle.load(f)
    val_messages = set(data_split["val_messages"])
    # print("val_messages: ", len(val_messages))
    # 3. Sub-select
    # val_df = df[df["message"].isin(val_messages)].copy()
    val_df = df.copy()
    # 4. Drop messages with category == -1
    val_df = val_df[val_df["category"] != -1].reset_index(drop=True)
    return val_df


# def init_vertex() -> GenerativeModel:
#     """Initialise Vertex AI and return a Gemini model handle."""
#     vertexai.init(project=PROJECT_ID, location=LOCATION)
#     model = GenerativeModel(GEMINI_MODEL_NAME)
#     return model
def init_vertex():
    # Requires GOOGLE_CLOUD_PROJECT & friends in env
    model = genai.Client(vertexai = use_vertexai, project = project, location=location).models
    return model

# _CATEGORY_KEY = {
#     0: "Loneliness / isolation",
#     1: "Self-harm ideation",
#     2: "Mentions self-harm in relation to others",
#     3: "Asks for help or resources",
#     4: "Recovery / success stories",
#     5: "Other mental-health related",
# }
# _CATEGORY_PROMPT = "\n".join(f"{k}: {v}" for k, v in _CATEGORY_KEY.items())


def classify_message(model, text: str) -> int:
    """
    Ask Gemini to return ONLY an integer 0-5.
    Falls back to -1 on parsing problems.
    """
    prompt = f"""
    Categorize this message into categorizes 0 through 5 based on these category descriptions:
    0: mentions being alone/lonely/preferring AI to humans
    1: mentions self-harm/suicide in general but not in relation to a person
    2: mentions self-harm suicide directly in relation to someone else
    3: Asks for resources to get help
    4: Non-urgent mention of self-harm or suicide
    5: Direct and urgent mention of self-harm or suicide
    Return only a number 0, 1, 2, 3, 4, or 5.
    Message: {text}
    """
    response = model.generate_content(
        model=GEMINI_MODEL_VER,
        contents=prompt
    )
    print("gemini response: ", response.text)
    # Extract first digit 0-5 from the response
    match = re.search(r"[0-5]", response.text)
    return int(match.group()) if match else -1 # -1 is fallback


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
    y_true = df["category"].astype(int)
    y_pred = df["predicted_category"].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in range(6)],
                             columns=[f"pred_{i}" for i in range(6)])
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv", index=True)
    print("\nConfusion matrix (saved to confusion_matrix.csv):\n")
    print(cm_df)

    # Optional: detailed metrics
    report = classification_report(
        y_true, y_pred, labels=list(range(6)), digits=3, output_dict=False
    )
    print("\nClassification report:\n")
    print(report)

    # ── Save mis-classified messages ───────────────────────────────────────────
    mismatches = df[df["predicted_category"] != df["category"]]
    mismatches.to_excel(OUTPUT_DIR / "mismatches.xlsx", index=False)
    print(f"\nSaved {len(mismatches):,} mis-classified messages to mismatches.xlsx")


if __name__ == "__main__":
    main()
