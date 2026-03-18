# Master's Report Draft

## Title Page (Replace With Your Institution Format)

**Title:** NeuroFusionGPT: Cross-Attention Fusion of EEG and ECG Signals for Stress Classification  
**Student Name:** [Your Name]  
**Registration Number:** [Your Registration Number]  
**Degree Program:** Master of [Program Name]  
**Department:** [Department Name]  
**Institution:** [University Name]  
**Supervisor:** [Supervisor Name]  
**Submission Date:** [Month Year]

---

## Abstract

Stress detection from physiological signals is a significant research area in affective computing, mental-health monitoring, and human-centered AI. This work presents **NeuroFusionGPT**, a multimodal deep learning framework that combines electroencephalogram (EEG) and electrocardiogram (ECG) signals through a cross-attention fusion mechanism for five-class stress classification (Calm, Mild Stress, Moderate Stress, High Stress, Severe Stress).

The proposed architecture contains three main modules: (1) an EEG Transformer encoder that captures inter-feature dependencies using positional encoding and multi-head self-attention, (2) an ECG multilayer perceptron (MLP) encoder for compact cardiac feature representation, and (3) a cross-attention fusion block that models interactions between brain and heart representations before final classification. To address class imbalance, the training pipeline uses Focal Loss with class weights, AdamW optimization, learning-rate scheduling, and early stopping based on macro F1-score.

The model was trained for up to 50 epochs with early stopping and selected based on validation macro F1-score. On the held-out test set, NeuroFusionGPT achieved **93.15% accuracy**, **91.33% balanced accuracy**, and **0.7826 macro F1-score**. Per-class analysis shows strong performance for Calm and Severe Stress classes, while minority classes remain comparatively challenging, indicating opportunities for improved class-balanced learning and domain-specific data augmentation.

Overall, this study demonstrates that multimodal fusion with cross-attention can provide robust stress classification performance and offers a practical path toward real-time wellness assistance systems, including optional natural-language feedback generation via large language model (LLM) integration.

**Keywords:** Multimodal learning, EEG, ECG, Stress detection, Cross-attention, Transformer, Focal loss, Affective computing

---

## Acknowledgment (Optional)

I would like to express my sincere gratitude to my supervisor, [Supervisor Name], for continuous guidance and support throughout this research. I also thank my department and peers for their encouragement and valuable feedback.

---

## Table of Contents

1. Introduction  
2. Problem Statement and Objectives  
3. Literature Review  
4. Proposed Methodology  
5. Experimental Setup  
6. Results and Discussion  
7. Conclusion and Future Work  
8. References  

---

## 1. Introduction

### 1.1 Background

Stress is a multidimensional physiological and psychological response that affects cognitive performance, emotional well-being, and long-term health outcomes. Conventional stress assessment techniques, including self-reported questionnaires and clinical observation, are often subjective, intermittent, and difficult to scale for continuous monitoring.

Biosignal-based machine learning offers an objective alternative. Among physiological modalities, EEG and ECG provide complementary information: EEG reflects neural dynamics and cognitive state, while ECG captures autonomic responses linked to stress. Integrating both modalities can improve robustness and representational richness compared with unimodal approaches.

### 1.2 Motivation

Most existing methods rely on single-modality pipelines or shallow fusion strategies (for example, simple concatenation of feature vectors). Such methods often fail to model complex cross-modal dependencies. This motivates a fusion framework that can explicitly learn interactions between EEG and ECG representations through attention-based mechanisms.

### 1.3 Scope of the Work

This work focuses on:
- Designing a multimodal architecture for EEG-ECG fusion.
- Training and evaluating the model for five-level stress classification.
- Handling class imbalance with suitable loss design and metrics.
- Providing an extensible inference pipeline for optional LLM-based wellness feedback.

---

## 2. Problem Statement and Objectives

### 2.1 Problem Statement

Given paired EEG and ECG feature vectors, predict the stress level for each sample across five classes:
1. Calm  
2. Mild Stress  
3. Moderate Stress  
4. High Stress  
5. Severe Stress

The primary technical challenge is learning meaningful cross-modal interactions under class imbalance and heterogeneous feature structures.

### 2.2 Objectives

1. Develop an EEG encoder using Transformer layers to capture non-local feature dependencies.  
2. Develop an ECG encoder using an MLP suitable for fixed-length feature vectors.  
3. Design a cross-attention fusion mechanism to combine EEG and ECG embeddings.  
4. Train the model with imbalance-aware optimization (Focal Loss + class weighting).  
5. Evaluate with class-sensitive metrics (macro F1 and balanced accuracy).  
6. Build a practical inference path with optional natural-language feedback generation.

---

## 3. Literature Review

### 3.1 Stress Detection From Physiological Signals

Prior work has explored stress recognition using ECG-derived heart rate variability features, EEG frequency-band analysis, galvanic skin response, and multimodal combinations. ECG-based approaches are relatively lightweight but can miss neural context; EEG-based approaches are expressive but may be sensitive to noise and domain shifts.

### 3.2 Deep Learning for Biosignals

Deep architectures such as CNNs, RNNs, and Transformers have improved representation learning over hand-engineered features. Transformers are especially useful for modeling long-range dependencies via attention. For tabular or fixed-dimensional physiological features, MLP-based encoders remain effective and computationally efficient.

### 3.3 Multimodal Fusion Strategies

Common fusion methods include early fusion (input concatenation), late fusion (decision-level integration), and intermediate fusion (shared latent space). Attention-based intermediate fusion can outperform concatenation by explicitly weighting cross-modal relevance.

### 3.4 Gap Identified

There remains a need for practical, end-to-end multimodal pipelines that:
- Explicitly model EEG-ECG interaction,
- Address class imbalance in stress classes,
- And produce deployment-ready outputs for downstream applications.

---

## 4. Proposed Methodology

### 4.1 Overview of NeuroFusionGPT

The proposed model has four stages:
1. **EEG Encoding** using a Transformer-based encoder.  
2. **ECG Encoding** using an MLP encoder.  
3. **Cross-Attention Fusion** to learn cross-modal dependencies.  
4. **Classification Head** for five-class stress prediction.

### 4.2 Input Representation

- **EEG input:** 178 features (treated as token sequence with positional encoding).  
- **ECG input:** 178 features (normalized with StandardScaler).  
- **Labels:** 5 classes, mapped to range 0-4.

### 4.3 EEG Transformer Encoder

The EEG branch projects each scalar feature to a `d_model=128` embedding and applies:
- Sinusoidal positional encoding,
- 4 Transformer encoder layers,
- 8 attention heads,
- GELU activation and dropout,
- Mean pooling and layer normalization.

This branch outputs a 128-dimensional EEG embedding.

### 4.4 ECG MLP Encoder

The ECG branch uses a feed-forward network with hidden dimensions `[256, 128]`, batch normalization, ReLU activation, and dropout. This branch outputs a 128-dimensional ECG embedding aligned with the EEG branch.

### 4.5 Cross-Attention Fusion

The fusion block performs cross-attention between EEG and ECG embeddings, followed by residual connections, normalization, and projection to a unified 128-dimensional representation. This allows the model to learn how heart and brain features inform each other instead of treating modalities independently.

### 4.6 Classification Head

The fused representation is passed through:
- Linear layer (128 to 256),
- ReLU + dropout,
- Output layer (256 to 5 logits),
followed by softmax for class probabilities.

### 4.7 Training Strategy

- **Optimizer:** AdamW  
- **Learning rate:** 0.0003  
- **Weight decay:** 0.01  
- **Batch size:** 128  
- **Max epochs:** 50  
- **Scheduler:** ReduceLROnPlateau  
- **Early stopping:** patience = 10, monitored on validation macro F1  
- **Loss:** Focal Loss (gamma = 2.0) with class weights

### 4.8 Evaluation Metrics

Primary metrics:
- Macro F1-score
- Balanced accuracy

Secondary metrics:
- Accuracy
- Weighted F1-score
- Per-class precision, recall, F1
- Confusion matrix

---

## 5. Experimental Setup

### 5.1 Data Preparation

The preprocessing pipeline includes:
- Loading EEG and ECG CSV files,
- Label alignment to a common 0-4 scale,
- Stratified train/validation/test splitting,
- Standard scaling for ECG features (fit on training data only),
- Class-weight computation from training labels.

### 5.2 Training Environment

Training is performed in Google Colab with optional GPU acceleration. Artifacts saved after training include:
- `best_fusion_model.pth`
- `scaler_eeg.pkl`, `scaler_ecg.pkl`
- `fusion_results.json`
- `training_history.json`
- Visualization figures (EDA, training curves, confusion matrix, ROC, etc.)

### 5.3 Reproducibility Configuration

The project uses a central configuration file (`configs/config.yaml`) that records:
- Data paths,
- Model hyperparameters,
- Training setup,
- Logging and checkpointing choices.

---

## 6. Results and Discussion

### 6.1 Training Summary

- Maximum epochs set to 50.  
- Early stopping triggered at epoch 37.  
- Best validation macro F1-score: **0.8010** at epoch 27.

### 6.2 Test Performance

| Metric | Value |
|---|---:|
| Test Loss | 0.2053 |
| Accuracy | 0.9315 |
| Balanced Accuracy | 0.9133 |
| Macro F1-score | 0.7826 |

### 6.3 Per-Class Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Calm | 0.9933 | 0.9289 | 0.9600 |
| Mild Stress | 0.3621 | 0.8147 | 0.5014 |
| Moderate Stress | 0.8206 | 0.9537 | 0.8821 |
| High Stress | 0.4704 | 0.8827 | 0.6137 |
| Severe Stress | 0.9269 | 0.9863 | 0.9557 |

### 6.4 Interpretation

1. **Strong overall classification quality** is indicated by high accuracy and balanced accuracy.  
2. **Macro F1 (0.7826)** is lower than accuracy, reflecting class-wise difficulty and imbalance sensitivity.  
3. Minority classes (Mild/High Stress) show high recall but lower precision, suggesting overprediction tendencies.  
4. Majority classes (Calm/Severe Stress) achieve very strong precision and F1, indicating stable class boundaries.

### 6.5 Discussion of Fusion Benefit

The cross-attention design provides a mechanism for learning modality interactions directly in latent space. Compared with pure concatenation baselines, this strategy is expected to improve representation quality by conditioning one modality on the other. (If required by your examiner, include explicit ablation tables: EEG-only, ECG-only, concat-fusion, cross-attention fusion.)

### 6.6 Limitations

1. Dataset-domain mismatch risk if modalities originate from different collection contexts.  
2. Pairing by index may not guarantee identical subject/time alignment.  
3. Class imbalance still affects precision for minority classes.  
4. Generalization to unseen populations requires additional external validation.

---

## 7. Conclusion and Future Work

This report presented NeuroFusionGPT, a multimodal cross-attention framework for stress classification using EEG and ECG signals. The model achieved strong quantitative results on held-out test data (accuracy 93.15%, balanced accuracy 91.33%, macro F1 0.7826), demonstrating that attention-based fusion can effectively leverage complementary brain and heart information.

Future work includes:
1. Collecting synchronized, stress-labeled multimodal datasets to reduce domain mismatch.  
2. Performing systematic ablation studies and statistical significance testing.  
3. Improving minority-class precision through augmentation and advanced rebalancing.  
4. Extending to real-time, privacy-aware deployment on edge or mobile platforms.  
5. Integrating calibrated uncertainty estimates before downstream LLM-generated guidance.

---

## 8. References (Replace With Proper Citation Style)

> Use your required style (IEEE/APA/Harvard). The examples below are placeholders.

1. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). *Focal Loss for Dense Object Detection*.  
2. Vaswani, A., et al. (2017). *Attention Is All You Need*.  
3. [Add EEG stress-detection papers used in your literature review.]  
4. [Add ECG/HRV stress-detection papers.]  
5. [Add multimodal fusion and affective computing references.]

---

## Appendix A: Suggested Viva/Defense Talking Points

1. Why EEG+ECG fusion is better than unimodal modeling.  
2. Why macro F1 and balanced accuracy are primary metrics under imbalance.  
3. How cross-attention differs from simple concatenation.  
4. Why Focal Loss improves minority class handling.  
5. Practical deployment path and ethical considerations (privacy, consent, non-clinical claims).

---

## Appendix B: Personalization Checklist Before Submission

- [ ] Replace all placeholders (name, supervisor, institution, dates).  
- [ ] Add institution-specific certificate/declaration pages.  
- [ ] Insert actual figures from `figures/` into the report body.  
- [ ] Add properly formatted citations from real papers.  
- [ ] Verify methodology text matches your final notebook version.  
- [ ] Add plagiarism declaration and AI-use statement if required by your university.
