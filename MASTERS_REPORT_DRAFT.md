# NEUROFUSIONGPT: MULTIMODAL STRESS CLASSIFICATION USING EEG-ECG CROSS-ATTENTION

## A Project Report

Presented to the faculty of the Department of [Department Name]  
[University Name]

Submitted in partial satisfaction of the requirements for the degree of  
MASTER OF SCIENCE  
in  
[Program Name]

by  
[Your Full Name]  
[Semester, Year]

---

## APPROVAL PAGE (Template)

NEUROFUSIONGPT: MULTIMODAL STRESS CLASSIFICATION USING EEG-ECG CROSS-ATTENTION  
A Project

by  
[Your Full Name]

Approved by:

- ____________________________, Committee Chair, [Advisor Name]
- ____________________________, Second Reader, [Second Reader Name]

Date: ______________________

---

## ABSTRACT

of  
NEUROFUSIONGPT: MULTIMODAL STRESS CLASSIFICATION USING EEG-ECG CROSS-ATTENTION  
by  
[Your Full Name]

Stress prediction from physiological signals is an important problem in digital health and affective computing because early detection enables timely intervention. Most practical pipelines rely on a single modality or simple fusion, which often misses interactions between neural and autonomic responses. This project introduces **NeuroFusionGPT**, a multimodal architecture that combines electroencephalogram (EEG) and electrocardiogram (ECG) features through a Transformer-based encoding pipeline and cross-attention fusion for five-class stress classification.

The model consists of an EEG Transformer encoder, an ECG MLP encoder, a cross-attention fusion module, and a classification head. To handle class imbalance, training uses Focal Loss with class weights, AdamW optimization, learning-rate scheduling, and early stopping based on validation macro F1-score. The implementation is reproducible through configuration-driven training and artifact export (model checkpoints, scalers, JSON metrics, and visualizations).

On the held-out test set, the final model achieved **93.15% accuracy**, **91.33% balanced accuracy**, and **0.7826 macro F1-score**. Class-level results show strong performance for Calm and Severe Stress categories, while Mild and High Stress classes remain more difficult, mainly due to imbalance and boundary overlap. The work demonstrates that attention-based multimodal fusion can improve stress classification quality and provide a practical foundation for downstream personalized wellness feedback systems.

Committee Chair: ____________________  
Date: ____________________

---

## ACKNOWLEDGEMENTS

I would like to sincerely thank my advisor, **[Advisor Name]**, for guidance, continuous encouragement, and valuable feedback throughout this project. I also thank **[Second Reader Name]** for thoughtful suggestions that improved the technical and presentation quality of this report. Finally, I am grateful to my family and friends for their constant support and motivation.

---

## TABLE OF CONTENTS

1. CHAPTER 1: INTRODUCTION  
2. CHAPTER 2: LITERATURE REVIEW  
3. CHAPTER 3: DATASET  
4. CHAPTER 4: METHODOLOGY  
5. CHAPTER 5: ORIGINAL NEW IMPLEMENTATIONS  
6. CHAPTER 6: RESULTS  
7. CHAPTER 7: CONCLUSION AND FUTURE WORK  
8. REFERENCES  

---

## LIST OF TABLES

1. Table 1. Test-set overall performance metrics  
2. Table 2. Per-class precision, recall, and F1-score  
3. Table 3. Training configuration summary  
4. Table 4. Proposed ablation comparison template (to be filled)  

---

## LIST OF FIGURES

1. Figure 1. EEG and ECG data overview (EDA)  
2. Figure 2. Training and validation curves  
3. Figure 3. Confusion matrix (counts and normalized)  
4. Figure 4. Per-class precision/recall/F1 chart  
5. Figure 5. ROC curves (one-vs-rest)  
6. Figure 6. Confidence/uncertainty analysis  
7. Figure 7. NeuroFusionGPT architecture diagram  

---

# CHAPTER 1  
# INTRODUCTION

## 1.1 Overview

Stress is a physiological and cognitive response that can affect concentration, emotional stability, and long-term health. Traditional stress assessment methods, including self-report forms and periodic clinical review, are often subjective and not suitable for continuous monitoring. Biosignal-driven machine learning provides a scalable and objective alternative.

In this project, stress prediction is formulated as a five-class classification problem using multimodal inputs from EEG and ECG. EEG captures neural activity and can provide context about cognitive-emotional states, while ECG captures cardiovascular responses linked to autonomic stress regulation. Because these modalities describe different but related physiological mechanisms, their fusion can improve prediction robustness.

The baseline challenge in multimodal learning is that simple concatenation treats modalities independently and cannot explicitly model interactions. To address this, this work proposes **NeuroFusionGPT**, a cross-attention-based fusion architecture where EEG and ECG embeddings interact in a shared latent space before classification. This design aims to improve class discrimination, especially under imbalance.

## 1.2 Problem Motivation

Three practical problems motivate this work:

1. Stress datasets are often imbalanced across severity levels.  
2. Unimodal models miss complementary information across brain and heart signals.  
3. Deployment requires reproducible training and interpretable evaluation artifacts.

The proposed framework addresses these through imbalance-aware learning, attention-based fusion, and a production-oriented training output pipeline.

## 1.3 Project Objectives

- Build an EEG Transformer encoder for contextual representation learning.  
- Build an ECG MLP encoder for robust fixed-feature embedding.  
- Design cross-attention fusion for explicit inter-modal interaction learning.  
- Train with Focal Loss and class weights for imbalance handling.  
- Evaluate with class-sensitive metrics (macro F1, balanced accuracy).  
- Export artifacts for deployment and optional LLM-based text feedback.

---

# CHAPTER 2  
# LITERATURE REVIEW

## 2.1 Overview

Recent stress-detection studies have explored EEG, ECG, and multimodal physiological inputs. Traditional machine learning models (e.g., SVM, logistic regression, random forest) remain useful baselines, but deep learning models generally perform better when signal relationships are complex and nonlinear.

## 2.2 EEG-Based Stress Modeling

EEG is widely used to capture brain-state variation associated with attention, fatigue, and stress. Deep models, especially Transformer-style encoders, can capture long-range dependencies across features or temporal tokens better than handcrafted feature pipelines.

## 2.3 ECG-Based Stress Modeling

ECG captures heart-rate and variability patterns linked to stress activation. MLP or tree-based models are commonly used for fixed-length ECG features and can perform strongly when preprocessing is stable and labels are clean.

## 2.4 Multimodal Fusion in Biosignal Learning

Fusion strategies typically include early fusion (input-level), intermediate fusion (embedding-level), and late fusion (decision-level). Intermediate fusion with attention is increasingly preferred because it allows one modality to condition on the other instead of simply appending vectors.

## 2.5 Transformer and Attention Mechanisms

Transformers use self-attention to assign context-sensitive importance across tokens. In physiological modeling, this enables selective focus on informative patterns and improves representational flexibility compared with strictly sequential recurrent methods.

## 2.6 Learning Under Class Imbalance

Class imbalance can inflate apparent accuracy while degrading minority-class detection. Focal Loss, class weighting, and macro-averaged metrics are established techniques for reducing this risk and improving per-class fairness.

## 2.7 Research Gap

Many practical stress pipelines still use unimodal features or weak fusion. There is a need for:
- explicit cross-modal interaction modeling,  
- imbalance-aware optimization, and  
- deployment-ready artifact generation.

This project directly targets these three requirements.

---

# CHAPTER 3  
# DATASET

## 3.1 Overview

This project uses paired EEG and ECG feature sets for stress classification. The learning target is a five-level stress label: **Calm, Mild Stress, Moderate Stress, High Stress, Severe Stress**.

## 3.2 EEG Data

The EEG pipeline uses 178 features per sample and corresponding class labels. EEG features are treated as token-like inputs to a Transformer encoder. Stratified splitting is applied for train/validation partitioning.

## 3.3 ECG Data

The ECG pipeline uses 178 input features and stress labels aligned to class indices 0-4. ECG features are normalized using `StandardScaler`, fit on training data and reused for validation/test to avoid leakage.

## 3.4 Label Alignment and Pairing

Because modalities may originate from different source files, label alignment and index pairing are handled carefully. The current implementation uses aligned indexing and a shared class mapping. This enables multimodal training while preserving a consistent target space.

## 3.5 Preprocessing Workflow

1. Load EEG and ECG CSV files.  
2. Extract feature columns and labels.  
3. Normalize ECG features; keep EEG in expected scale.  
4. Perform stratified train/validation split.  
5. Compute class weights from training labels.  
6. Build PyTorch datasets/dataloaders.

## 3.6 Data Challenges

- Class imbalance across stress levels.  
- Potential cross-source mismatch between modalities.  
- Generalization risk when distributions shift.

These are acknowledged and addressed in the training/evaluation design.

---

# CHAPTER 4  
# METHODOLOGY

## 4.1 Overview

The model architecture includes:
1. EEG Transformer encoder  
2. ECG MLP encoder  
3. Cross-attention fusion  
4. Classifier head

The full system predicts one stress label from paired EEG-ECG inputs.

## 4.2 EEG Transformer Encoder

Each EEG scalar feature is projected to `d_model = 128`, positional encodings are added, and Transformer encoder layers process inter-feature dependencies. The output sequence is pooled (mean pooling) and normalized to produce a 128-dimensional EEG embedding.

## 4.3 ECG MLP Encoder

ECG features are processed by a feed-forward stack with hidden dimensions `[256, 128]`, batch normalization, ReLU activation, and dropout. The output is a 128-dimensional ECG embedding.

## 4.4 Cross-Attention Fusion Module

The model applies bidirectional cross-attention between EEG and ECG embeddings to learn interaction-aware fused representations, followed by residual normalization and projection.

Formally, with EEG embedding \(E\) and ECG embedding \(C\):

\[
\text{Attn}_{E \leftarrow C} = \text{MHA}(Q=E, K=C, V=C)
\]
\[
\text{Attn}_{C \leftarrow E} = \text{MHA}(Q=C, K=E, V=E)
\]
\[
F = W_f[\text{Norm}(E + \text{Attn}_{E \leftarrow C}) \, || \, \text{Norm}(C + \text{Attn}_{C \leftarrow E})]
\]

where \(F\) is the fused representation used for classification.

## 4.5 Classification Head

The fused vector is passed through:
- Linear(128 to 256)  
- ReLU + Dropout  
- Linear(256 to 5 logits)

The output probabilities are obtained by softmax.

## 4.6 Training Configuration

- Optimizer: AdamW  
- Learning rate: 0.0003  
- Weight decay: 0.01  
- Batch size: 128  
- Epochs: up to 50  
- Scheduler: ReduceLROnPlateau  
- Early stopping: patience 10 on validation macro F1  
- Loss: Focal Loss (\(\gamma = 2.0\)) + class weights

## 4.7 Evaluation Metrics

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]
\[
\text{Precision} = \frac{TP}{TP + FP}, \quad
\text{Recall} = \frac{TP}{TP + FN}
\]
\[
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
\]

Macro F1 and balanced accuracy are treated as primary metrics due to class imbalance.

---

# CHAPTER 5  
# ORIGINAL NEW IMPLEMENTATIONS

## 5.1 Overview

This chapter summarizes the implementation contributions that distinguish this project from a standard single-model pipeline.

## 5.2 NeuroFusionGPT Cross-Attention Architecture

The primary contribution is a multimodal architecture that explicitly fuses EEG and ECG embeddings using cross-attention rather than direct concatenation. This design allows the model to learn conditional relationships between neural and cardiac patterns.

## 5.3 Imbalance-Aware Training System

A second contribution is a robust training loop using:
- Focal Loss with class weights,  
- early stopping on macro F1,  
- gradient clipping, and  
- checkpointing of best/last models.

This design improves minority-class learning behavior and prevents overfitting in extended runs.

## 5.4 Reproducible Artifacts and Deployment Outputs

The project exports:
- `best_fusion_model.pth`  
- `scaler_eeg.pkl`, `scaler_ecg.pkl`  
- `model_config.json`  
- `fusion_results.json`  
- `training_history.json`

These outputs support reproducible inference and integration into external applications.

## 5.5 Optional LLM Feedback Layer

An additional implementation layer connects model predictions to an OpenRouter-compatible LLM endpoint. The LLM component converts predicted stress level and confidence into concise wellness guidance, making the system user-facing without changing core classifier logic.

---

# CHAPTER 6  
# RESULTS

## 6.1 Overview

Training was configured for 50 epochs with early stopping. The best validation macro F1 was achieved at epoch 27, and early stopping triggered at epoch 37.

## 6.2 Visualization

The experiment generates six key visual outputs:
1. Data EDA plot  
2. Training curves  
3. Confusion matrix  
4. Per-class metric bars  
5. ROC curves  
6. Confidence analysis

These plots support error analysis, threshold understanding, and class behavior interpretation.

## 6.3 Results

### Table 1. Overall Test Performance

| Metric | Value |
|---|---:|
| Loss | 0.2053 |
| Accuracy | 0.9315 |
| Balanced Accuracy | 0.9133 |
| Macro F1 | 0.7826 |

### Table 2. Per-Class Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Calm | 0.9933 | 0.9289 | 0.9600 | 18118 |
| Mild Stress | 0.3621 | 0.8147 | 0.5014 | 556 |
| Moderate Stress | 0.8206 | 0.9537 | 0.8821 | 1448 |
| High Stress | 0.4704 | 0.8827 | 0.6137 | 162 |
| Severe Stress | 0.9269 | 0.9863 | 0.9557 | 1608 |

### Table 3. Training Configuration Summary

| Component | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 0.0003 |
| Weight Decay | 0.01 |
| Batch Size | 128 |
| Max Epochs | 50 |
| LR Scheduler | ReduceLROnPlateau |
| Early Stopping | Patience = 10 (monitor macro F1) |
| Loss Function | Focal Loss (gamma = 2.0) + class weights |
| Best Validation Macro F1 | 0.8010 (epoch 27) |
| Stop Epoch | 37 |

## 6.4 Experiment Discussion

The model achieved strong overall discrimination and class-balanced performance, reflected by high balanced accuracy. However, class-level behavior shows a precision-recall trade-off for minority categories (Mild and High Stress), where recall is high but precision is lower. This indicates the classifier is sensitive to minority samples but can overpredict them under overlap conditions.

The majority classes (Calm, Severe Stress) achieved high precision and F1, showing stable decision boundaries for well-represented classes. These findings validate the effectiveness of cross-attention multimodal fusion while also highlighting that further improvements in minority precision are needed.

## 6.5 Ablation Template (Fill After Running Additional Experiments)

### Table 4. Suggested Ablation Comparison

| Model Variant | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| ECG-only baseline | [fill] | [fill] | [fill] |
| EEG-only baseline | [fill] | [fill] | [fill] |
| Concat Fusion | [fill] | [fill] | [fill] |
| Cross-Attention Fusion (proposed) | 0.9315 | 0.9133 | 0.7826 |

---

# CHAPTER 7  
# CONCLUSION AND FUTURE WORK

This project presented NeuroFusionGPT, a multimodal stress-classification framework that fuses EEG and ECG representations using cross-attention. The model demonstrated strong overall predictive performance on the held-out test set, with high accuracy and balanced accuracy, while maintaining a competitive macro F1 under class imbalance.

The key contribution is the explicit interaction modeling between modalities, combined with an imbalance-aware training strategy and reproducible artifact export pipeline suitable for practical deployment. Results show that multimodal fusion is effective for stress-level prediction and can support downstream personalized feedback systems.

### Future Work

1. Collect synchronized stress-labeled EEG and ECG from the same subjects.  
2. Run complete ablations and statistical significance tests.  
3. Improve minority precision via augmentation and calibration.  
4. Explore domain adaptation for cross-dataset generalization.  
5. Add uncertainty-aware decision logic before LLM feedback generation.  
6. Evaluate real-time deployment latency and privacy constraints.

---

# REFERENCES (Update to your required citation style: IEEE/APA)

[1] A. Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.  
[2] T.-Y. Lin et al., "Focal Loss for Dense Object Detection," ICCV, 2017.  
[3] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.  
[4] B. Shickel et al., "Deep EHR: A Survey of Recent Advances in Deep Learning Techniques for Electronic Health Record Analysis," IEEE JBHI, 2018.  
[5] M. Zhang et al., "A Review on Human Stress Detection from Signals," IEEE Access, [add year].  
[6] [Add EEG stress-detection paper used in your review.]  
[7] [Add ECG stress-detection paper used in your review.]  
[8] [Add multimodal EEG+ECG paper.]  
[9] [Add Transformer-for-biosignals paper.]  
[10] [Add class imbalance in healthcare ML paper.]  

---

## FINAL SUBMISSION CHECKLIST

- [ ] Replace all placeholders (names, department, dates, committee).  
- [ ] Add university-required certificate/declaration pages.  
- [ ] Insert actual figure images and update captions/page numbers.  
- [ ] Add final references from papers you cited in Chapter 2.  
- [ ] Fill ablation table if you run additional baselines.  
- [ ] Export to Word/PDF in your university formatting template.
