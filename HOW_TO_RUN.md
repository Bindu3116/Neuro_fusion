# How to Run the Project Locally

This guide explains how to set up your environment, get the dataset and trained model from Colab, place them in the right folders, and run the project on your machine.

---

## 1. Create the Python Environment

Training and inference were done in **Google Colab**. To run the project locally, use a virtual environment and install dependencies.

### Option A: venv (recommended)

**Windows (PowerShell):**
```powershell
cd E:\project\bindu-project
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
cd E:\project\bindu-project
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
cd /path/to/bindu-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option B: conda

```bash
cd E:\project\bindu-project
conda create -n bindu python=3.10 -y
conda activate bindu
pip install -r requirements.txt
```

After this, your environment is ready. Activate the same environment whenever you work on the project (`.\venv\Scripts\Activate.ps1` on Windows with venv, or `conda activate bindu` with conda).

---

## 2. Get the Dataset and Model from Colab

The model and datasets are **not** in this repository. They were produced in **Google Colab** during training. You need to download them from Colab (or from Google Drive where Colab saved them) and place them in this project.

### Where training runs

- Training is done in the Colab notebook (e.g. `notebook/neurofusiongpt_training.ipynb` or the fusion training notebook).
- Data in Colab is read from **Google Drive**: `My Drive/datasets/bindu/`.
- Trained outputs are saved to **Google Drive**: `My Drive/models/bindu/output/`.

### What to download

1. **From Colab / Google Drive – datasets**
   - From the Colab notebook or from **My Drive/datasets/bindu/** download:
     - `eeg_data.csv` (or `eeg_train.csv` / `eeg_test.csv`, as used in the notebook)
     - `ecg_train.csv`, `ecg_test.csv` (or the exact filenames your notebook uses)

2. **From Colab / Google Drive – model and artifacts**
   - From **My Drive/models/bindu/output/** (or the output path your notebook uses) download:
     - `best_fusion_model.pth` (trained model weights)
     - `scaler_eeg.pkl`, `scaler_ecg.pkl` (preprocessing scalers)
     - `model_config.json` (model configuration)
     - Optionally: `fusion_results.json`, `training_history.json`, and the `figures/` folder

Use the same Colab notebook / Drive folder where you (or your team) actually ran training. If someone shares a Colab link, open it, run or re-run the notebook, then download from the paths above (or from the notebook’s “Files” / “Mount Drive” view).

---

## 3. Place Files in This Project

After downloading, put everything in the right folders so the code can find them.

### Datasets

- Put all CSV dataset files inside the **`datasets/`** folder:

```
bindu-project/
  datasets/
    ecg_train.csv
    ecg_test.csv
    eeg_data.csv
    (or eeg_train.csv, eeg_test.csv – whatever the notebook uses)
```

### Model and related files

- Put the trained model and artifacts inside the **`models/`** folder (same structure as in Colab output if possible):

```
bindu-project/
  models/
    best_fusion_model.pth
    scaler_eeg.pkl
    scaler_ecg.pkl
    model_config.json
    fusion_results.json      # optional
    training_history.json    # optional
    figures/                 # optional
      data_eda.png
      training_curves.png
      ...
```

Paths used in code (e.g. in `configs/config.yaml` or scripts) may assume:
- Data under `datasets/`
- Model under `models/`

Adjust config or script paths if you use a different layout.

---

## 4. Environment Variables (Optional)

For LLM integration (e.g. OpenRouter), the project can use a `.env` file in the project root. Create it if needed:

```text
OPENROUTER_API_KEY=your_api_key_here
```

Do not commit `.env`; it is already in `.gitignore`.

---

## 5. Run the Project

- **Training:** Intended to be run in Colab using the notebook. Use the same notebook and Drive paths as in step 2.
- **Inference / scripts:** From the project root with your environment activated:

```powershell
# Example: run prediction and feedback script
python scripts/run_predict_and_feedback.py
```

Use the same Python environment you created in step 1. If scripts expect specific paths for `models/` or `datasets/`, ensure they match where you placed the files in step 3.

---

## Summary

| Step | Action |
|------|--------|
| 1 | Create env: `python -m venv venv` (or conda), then `pip install -r requirements.txt` |
| 2 | Download dataset and model from the Colab notebook / Google Drive (datasets and `models/.../output/`) |
| 3 | Put CSVs in **`datasets/`**, and model/scalers/config in **`models/`** |
| 4 | Add `.env` with `OPENROUTER_API_KEY` if you use LLM features |
| 5 | Activate the env and run scripts (e.g. `python scripts/run_predict_and_feedback.py`) |

If you have a specific Colab link to share in this doc, add it in section 2 so others know exactly where to download the dataset and model.
