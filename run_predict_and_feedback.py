#!/usr/bin/env python3
"""
NeuroFusionGPT: Load trained model, predict stress from EEG+ECG row, get LLM feedback.

Usage:
  1. Set OPENROUTER_API_KEY in environment (or .env with python-dotenv).
  2. Run:
     python run_predict_and_feedback.py --eeg-row 100 --ecg-row 100
     python run_predict_and_feedback.py --eeg-row 0 --ecg-row 0 --no-llm   # prediction only

Data: expects eeg_data.csv (or eed_data.csv) and ecg_train.csv in project root or --data-dir.
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import sys
import json
import argparse
import numpy as np
import torch
import joblib

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.neurofusiongpt import NeuroFusionGPT


def load_config(models_dir: str) -> dict:
    path = os.path.join(models_dir, "model_config.json")
    with open(path, "r") as f:
        return json.load(f)


def load_model_and_scalers(models_dir: str, device: str = "cpu"):
    config = load_config(models_dir)
    model_path = os.path.join(models_dir, config["model_path"])
    scaler_eeg_path = os.path.join(models_dir, config["scaler_eeg_path"])
    scaler_ecg_path = os.path.join(models_dir, config["scaler_ecg_path"])

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = NeuroFusionGPT(
        eeg_dim=config["eeg_features"],
        ecg_dim=config["ecg_features"],
        d_model=config["d_model"],
        num_classes=config["num_classes"],
    )
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    scaler_eeg = joblib.load(scaler_eeg_path)
    scaler_ecg = joblib.load(scaler_ecg_path)

    return model, scaler_eeg, scaler_ecg, config


def load_eeg_row(data_dir: str, row: int) -> np.ndarray:
    for name in ("eeg_data.csv", "eed_data.csv"):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            import pandas as pd
            df = pd.read_csv(path)
            feat_cols = [c for c in df.columns if c != "y"]
            X = df[feat_cols].values.astype(np.float32)
            if row < 0 or row >= len(X):
                raise IndexError(f"EEG row {row} out of range [0, {len(X)-1}]")
            return X[row]
    raise FileNotFoundError(f"No eeg_data.csv or eed_data.csv in {data_dir}")


def load_ecg_row(data_dir: str, row: int) -> np.ndarray:
    path = os.path.join(data_dir, "ecg_train.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ecg_train.csv not found in {data_dir}")
    arr = np.loadtxt(path, delimiter=",", max_rows=row + 1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # Last column is label, rest are features
    n_features = 187
    row_data = arr[row, :n_features].astype(np.float32)
    return row_data


def predict(
    model: torch.nn.Module,
    scaler_eeg,
    scaler_ecg,
    config: dict,
    eeg_raw: np.ndarray,
    ecg_raw: np.ndarray,
    device: str = "cpu",
) -> dict:
    eeg = scaler_eeg.transform(eeg_raw.reshape(1, -1)).astype(np.float32)
    ecg = scaler_ecg.transform(ecg_raw.reshape(1, -1)).astype(np.float32)
    eeg_t = torch.from_numpy(eeg).to(device)
    ecg_t = torch.from_numpy(ecg).to(device)

    with torch.no_grad():
        logits = model(eeg_t, ecg_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    class_names = config["class_names"]
    return {
        "predicted_class": pred_class,
        "predicted_label": class_names[pred_class],
        "confidence": float(probs[pred_class]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def main():
    parser = argparse.ArgumentParser(
        description="NeuroFusionGPT: predict stress and get LLM feedback"
    )
    parser.add_argument(
        "--models-dir",
        default=os.path.join(os.path.dirname(__file__), "models"),
        help="Directory containing model_config.json, .pth, and scalers",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.dirname(__file__),
        help="Directory containing eeg_data.csv and ecg_train.csv",
    )
    parser.add_argument("--eeg-row", type=int, default=0, help="EEG sample row index")
    parser.add_argument("--ecg-row", type=int, default=0, help="ECG sample row index")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Only run model prediction, do not call OpenRouter",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model",
    )
    args = parser.parse_args()

    print("Loading model and scalers...")
    model, scaler_eeg, scaler_ecg, config = load_model_and_scalers(
        args.models_dir, args.device
    )
    print("Loading EEG and ECG rows...")
    eeg_raw = load_eeg_row(args.data_dir, args.eeg_row)
    ecg_raw = load_ecg_row(args.data_dir, args.ecg_row)

    print("Running prediction...")
    result = predict(
        model, scaler_eeg, scaler_ecg, config,
        eeg_raw, ecg_raw, args.device,
    )

    print("\n" + "=" * 60)
    print("STRESS PREDICTION")
    print("=" * 60)
    print(f"Predicted: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("Probabilities:")
    for label, p in result["probabilities"].items():
        print(f"  {label}: {p:.4f}")
    print("=" * 60)

    if not args.no_llm:
        try:
            from src.inference.openrouter_llm import generate_feedback
            feedback = generate_feedback(
                result["predicted_label"],
                result["confidence"],
                result["probabilities"],
            )
            print("\nWELLNESS FEEDBACK (OpenRouter / Grok)")
            print("-" * 60)
            print(feedback)
            print("-" * 60)
        except Exception as e:
            print(f"\nLLM feedback failed: {e}")
            print("Set OPENROUTER_API_KEY or use --no-llm for prediction only.")

    return result


if __name__ == "__main__":
    main()
