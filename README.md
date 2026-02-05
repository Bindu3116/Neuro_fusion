# NeuroFusionGPT: Multimodal Transformer for Brain-Signal Understanding

A deep learning framework with **cross-attention fusion** of EEG and ECG for stress detection. **Designed to run in Google Colab** with data on Google Drive.

## Quick Start

**Use the fusion notebook**: `notebook/neurofusiongpt_fusion_training.ipynb`

This implements the complete proposal architecture:
- Cross-attention fusion between EEG and ECG
- Single stress-level prediction output
- Ready for LLM integration at application level
- Comprehensive visualizations (6 plots)

**Read**: [`HOW_TO_RUN.md`](HOW_TO_RUN.md) for setup, downloading artifacts from Colab, and running the project locally.

## Run in Google Colab

### Fusion Model (Recommended - Implements Proposal)

1. **Upload datasets to Google Drive**:
   - Create folder: **My Drive/datasets/bindu/**
   - Upload **eeg_data.csv** (or eed_data.csv), **ecg_train.csv**, **ecg_test.csv**

2. **Open the fusion notebook in Colab:**
   - File: **notebook/neurofusiongpt_fusion_training.ipynb**
   - Runtime → Change runtime type → GPU (optional, faster)

3. **Run all cells.** The notebook will:
   - Train the fusion model (EEG Transformer + ECG MLP + Cross-Attention)
   - Generate **6 comprehensive visualizations**
   - Save model, scalers, and config to **My Drive/models/bindu/output/**

4. **Download trained artifacts** for your application:
   - `best_fusion_model.pth` (model checkpoint)
   - `scaler_eeg.pkl`, `scaler_ecg.pkl` (preprocessing)
   - `model_config.json` (architecture info)
   - All figures/ (visualizations)

**Paths:**
- Data: `/content/drive/MyDrive/datasets/bindu/`
- Output: `/content/drive/MyDrive/models/bindu/output/`

---

## Datasets

| Dataset   | Source | Features | Classes | Labels |
|-----------|--------|----------|---------|--------|
| **EEG** | BEED (Bangalore EEG Epilepsy) | 16 channels | 4 | Seizure types |
| **ECG** | Stress dataset | 187 features | 5 | Stress levels (0-4) |

**Fusion Target**: ECG stress labels (Calm → Severe Stress)  
**Current**: EEG provides neurological context; ECG provides stress labels  
**When you get stress EEG**: Both will predict stress (same target)

---

## Project structure

```
bindu-project/
├── notebook/
│   └── neurofusiongpt_training.ipynb   # Run this in Colab
├── src/                                 # Core code (used by notebook logic)
│   ├── data/
│   ├── models/
│   ├── training/
│   └── inference/
├── configs/
│   └── config.yaml
├── DATA_UPLOAD_GUIDE.md                 # What to upload and where
└── README.md
```

---

## Which data to upload and which file to run

| What | Where / Which file |
|------|---------------------|
| **Datasets to upload** | **eeg_train.csv**, **eeg_test.csv** |
| **Upload location** | **My Drive/datasets/bindu/** |
| **File to run** | **notebook/neurofusiongpt_training.ipynb** (in Google Colab) |
| **Outputs** | **My Drive/models/bindu/** (model + figures + JSONs) |

Full details: **[DATA_UPLOAD_GUIDE.md](DATA_UPLOAD_GUIDE.md)**

---

## Outputs

After training the **fusion model**, in **My Drive/models/bindu/output/** you get:

### Model Files (for application):
- **best_fusion_model.pth** – Trained fusion model (~2 MB)
- **scaler_eeg.pkl**, **scaler_ecg.pkl** – Preprocessing scalers
- **model_config.json** – Architecture configuration

### Results:
- **fusion_results.json** – Test metrics, confusion matrix, per-class metrics
- **training_history.json** – Epoch-by-epoch training log

### Visualizations (6 plots):
- **figures/data_eda.png** – Dataset exploration
- **figures/training_curves.png** – Loss, accuracy, F1 over epochs
- **figures/confusion_matrix.png** – Prediction matrix (raw + normalized)
- **figures/per_class_metrics.png** – Precision/Recall/F1 per stress level
- **figures/roc_curves.png** – ROC curves for all classes
- **figures/confidence_analysis.png** – Model uncertainty patterns

---

## Architecture

```
EEG (16 channels) → Transformer Encoder → 128-dim embedding
                                                ↓
                                        Cross-Attention ← Fusion!
                                                ↑
ECG (187 features) → MLP Encoder ────→ 128-dim embedding
                                                ↓
                                        Fused Embedding
                                                ↓
                                        Classifier → P(stress)
```

**Key Feature**: Bidirectional cross-attention learns inter-modal dependencies (brain ↔ heart).

## Application Integration

After training, use the model in your app:

```python
# Load model + scalers
model = NeuroFusionGPT(...)
model.load_state_dict(torch.load('best_fusion_model.pth'))
scaler_eeg = joblib.load('scaler_eeg.pkl')
scaler_ecg = joblib.load('scaler_ecg.pkl')

# User selects row (e.g., 1-20000)
eeg, ecg = load_data(row_number)
prediction = model.predict(eeg, ecg)

# Send to LLM
prompt = f"Stress: {prediction['label']}, Confidence: {prediction['confidence']}"
advice = llm_api.generate(prompt)
display(advice)
```

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for environment setup, where to place datasets and models, and how to run the project.

## Requirements

- **Colab**: All libraries pre-installed
- **Local**: `pip install -r requirements.txt`

Main dependencies: `torch`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `tqdm`
