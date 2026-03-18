# NEUROFUSIONGPT: MULTIMODAL STRESS CLASSIFICATION USING EEG AND ECG WITH CROSS-ATTENTION FUSION

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

## APPROVAL PAGE

**NEUROFUSIONGPT: MULTIMODAL STRESS CLASSIFICATION USING EEG AND ECG WITH CROSS-ATTENTION FUSION**  
A Project

by  
[Your Full Name]

Approved by:

- ________________________________, Committee Chair, [Advisor Name]  
- ________________________________, Second Reader, [Second Reader Name]

Date: ____________________

---

## STUDENT CERTIFICATION (TEMPLATE)

Student: [Your Full Name]

I certify that this student has met the requirements for format contained in the University format manual, and this project is suitable for electronic submission to the library and credit is to be awarded for the project.

______________________, Graduate Coordinator  
Department of [Department Name]  
Date: ____________________

---

## ABSTRACT

of  
**NEUROFUSIONGPT: MULTIMODAL STRESS CLASSIFICATION USING EEG AND ECG WITH CROSS-ATTENTION FUSION**  
by  
[Your Full Name]

Stress detection from physiological signals is an active research area in healthcare AI, affective computing, and wearable intelligence because timely stress recognition supports early intervention and improved wellbeing. Most practical stress prediction systems are unimodal (EEG-only or ECG-only) or use basic concatenation-based fusion, which limits their ability to learn interactions between neural and cardiovascular dynamics. This project proposes **NeuroFusionGPT**, a multimodal deep learning framework that combines electroencephalogram (EEG) and electrocardiogram (ECG) information using cross-attention fusion for five-class stress classification (Calm, Mild Stress, Moderate Stress, High Stress, Severe Stress).

The proposed pipeline includes an EEG Transformer encoder, an ECG MLP encoder, a bidirectional cross-attention fusion module, and a classifier head. The training strategy is explicitly designed for imbalanced classes through Focal Loss and class weighting, with AdamW optimization, learning-rate scheduling, gradient clipping, and early stopping based on validation macro F1-score. The implementation supports reproducibility through configuration-driven training and export of model artifacts, preprocessing scalers, and comprehensive evaluation outputs.

The best model was obtained at validation macro F1 = 0.8010 (epoch 27). On the held-out test set, NeuroFusionGPT achieved **accuracy = 0.9315**, **balanced accuracy = 0.9133**, and **macro F1 = 0.7826**. Per-class analysis showed high performance for Calm and Severe Stress classes, while minority classes (Mild Stress and High Stress) remained more challenging, primarily due to class imbalance and boundary overlap. These findings demonstrate that attention-based multimodal fusion is effective for robust stress classification and can serve as a foundation for practical decision-support systems, including optional natural-language wellness feedback generation using an LLM interface.

Committee Chair: ____________________  
Date: ____________________

---

## ACKNOWLEDGEMENTS

I would like to express my sincere gratitude to my advisor, **[Advisor Name]**, for constant guidance, technical mentorship, and encouragement throughout this project. I also thank **[Second Reader Name]** for valuable comments and feedback that improved the quality of this work.

I am deeply grateful to my family and friends for their support and motivation during my graduate studies. Their encouragement made this journey possible.

---

## TABLE OF CONTENTS

1. CHAPTER 1 - INTRODUCTION  
1.1 Overview  
1.2 Problem Statement  
1.3 Motivation  
1.4 Objectives  
1.5 Contributions  
1.6 Report Organization  

2. CHAPTER 2 - LITERATURE REVIEW  
2.1 Overview  
2.2 Stress Detection from Physiological Signals  
2.3 EEG-Based Modeling  
2.4 ECG-Based Modeling  
2.5 Multimodal Fusion Approaches  
2.6 Transformer Architectures in Biosignal Learning  
2.7 Class Imbalance and Evaluation  
2.8 Research Gap  

3. CHAPTER 3 - DATASET AND PREPROCESSING  
3.1 Overview  
3.2 Data Sources and Label Space  
3.3 EEG Data Pipeline  
3.4 ECG Data Pipeline  
3.5 Label Alignment and Pairing  
3.6 Train/Validation/Test Strategy  
3.7 Class Imbalance Profile  
3.8 Preprocessing Summary  

4. CHAPTER 4 - METHODOLOGY  
4.1 Overview  
4.2 Problem Formulation  
4.3 EEG Transformer Encoder  
4.4 ECG MLP Encoder  
4.5 Cross-Attention Fusion  
4.6 Classification Head  
4.7 Loss Function and Optimization  
4.8 Regularization and Early Stopping  
4.9 Evaluation Metrics  

5. CHAPTER 5 - ORIGINAL NEW IMPLEMENTATIONS  
5.1 Overview  
5.2 NeuroFusionGPT Architecture Implementation  
5.3 Imbalance-Aware Training Pipeline  
5.4 Experiment Tracking and Artifact Export  
5.5 Visualization and Error Analysis Pipeline  
5.6 Optional LLM Feedback Integration  

6. CHAPTER 6 - RESULTS AND DISCUSSION  
6.1 Overview  
6.2 Training Dynamics  
6.3 Quantitative Results  
6.4 Class-wise Analysis  
6.5 Discussion of Strengths and Weaknesses  
6.6 Threats to Validity  
6.7 Suggested Ablation and Extension Experiments  

7. CHAPTER 7 - CONCLUSION AND FUTURE WORK  
7.1 Conclusion  
7.2 Future Work  

REFERENCES  
APPENDIX A - VIVA QUESTIONS AND ANSWERS  
APPENDIX B - SUBMISSION CHECKLIST  

---

## LIST OF TABLES

Table 1. NeuroFusionGPT training configuration  
Table 2. Test-set overall metrics  
Table 3. Per-class performance report  
Table 4. Suggested ablation template  
Table 5. Risk and limitation summary  

---

## LIST OF FIGURES

Figure 1. Proposed NeuroFusionGPT architecture  
Figure 2. Data processing and training workflow  
Figure 3. Training and validation performance curves  
Figure 4. Confusion matrix visualization  
Figure 5. Per-class precision/recall/F1 bars  
Figure 6. ROC curves (one-vs-rest)  
Figure 7. Confidence and uncertainty analysis  

---

# CHAPTER 1
# INTRODUCTION

## 1.1 Overview

Stress is a complex psycho-physiological condition associated with cognitive load, emotional imbalance, and autonomic nervous system activation. Chronic stress can affect mental health, sleep quality, cardiovascular stability, and overall productivity. Therefore, automatic stress detection has become a key problem in healthcare AI and human-centered computing.

Traditional stress assessment methods include questionnaires and periodic clinical evaluations. Although clinically useful, these approaches are limited by subjectivity, irregular sampling, and poor scalability for continuous monitoring. In contrast, physiological signals can provide objective and continuous indicators of stress state.

Among physiological modalities, **EEG** reflects brain activity and cognitive-emotional processes, while **ECG** reflects cardiac activity and autonomic regulation. Since stress affects both brain and heart systems, combining EEG and ECG is a natural multimodal strategy. However, fusion design is critical: naive feature concatenation often fails to model interaction patterns explicitly.

To address this challenge, this project proposes **NeuroFusionGPT**, a deep multimodal framework that combines:

1. EEG Transformer encoding for contextual feature learning,  
2. ECG MLP encoding for robust cardiac representation, and  
3. Cross-attention fusion for explicit inter-modal dependency learning.

## 1.2 Problem Statement

Given paired EEG and ECG feature vectors, predict stress level across five classes:

0. Calm  
1. Mild Stress  
2. Moderate Stress  
3. High Stress  
4. Severe Stress

The model must remain robust under class imbalance and should provide class-sensitive evaluation outcomes suitable for practical use.

## 1.3 Motivation

Three major practical motivations drive this work:

- **Need for robust multimodal learning:** Unimodal models are sensitive to missing context and distribution shifts.  
- **Need for interaction-aware fusion:** Stress-related physiology is inherently cross-modal, so explicit interaction modeling is preferable to independent processing.  
- **Need for deployment readiness:** Real systems require reproducible preprocessing, checkpoints, metrics, and interpretable outputs.

## 1.4 Objectives

The project objectives are:

1. Build a Transformer-based EEG encoder to capture high-order feature dependencies.  
2. Build an MLP-based ECG encoder for compact cardiac embeddings.  
3. Fuse EEG and ECG through cross-attention.  
4. Train with imbalance-aware loss and early stopping.  
5. Evaluate using macro-sensitive and balanced metrics.  
6. Provide a practical output pipeline for downstream applications.

## 1.5 Contributions

The key contributions of this project are:

- A cross-attention-based multimodal stress classifier (NeuroFusionGPT).  
- A complete imbalance-aware training setup (Focal Loss + class weights).  
- A reproducible artifact pipeline (model, scalers, configs, JSON metrics, figures).  
- Practical interpretation through class-wise analysis and confusion behavior.  
- Optional LLM interface for user-facing feedback generation.

## 1.6 Report Organization

- Chapter 2 reviews prior research on stress modeling and multimodal learning.  
- Chapter 3 describes dataset handling and preprocessing design.  
- Chapter 4 presents architecture and training methodology.  
- Chapter 5 details original implementation features.  
- Chapter 6 reports experimental results and analysis.  
- Chapter 7 concludes and outlines future extensions.

---

# CHAPTER 2
# LITERATURE REVIEW

## 2.1 Overview

Stress detection has evolved from rule-based and handcrafted-feature systems to deep learning methods that can learn discriminative patterns directly from multimodal signals. This chapter reviews key areas relevant to the proposed work.

## 2.2 Stress Detection from Physiological Signals

Early stress detection systems relied on handcrafted signal features and shallow classifiers. Common signals include EEG, ECG, galvanic skin response (GSR), respiration, and skin temperature. These systems demonstrated feasibility but were often limited by feature engineering effort and low transferability across datasets.

Recent approaches use neural networks for end-to-end or representation learning. These methods reduce manual feature dependency and can capture nonlinear interactions between physiological variables.

## 2.3 EEG-Based Modeling

EEG provides insights into cortical activity and is frequently used in emotion, cognitive-load, and stress studies. Traditional methods rely on power spectral density and handcrafted band features. Deep methods (CNN, RNN, Transformer) can learn richer representations.

Transformer-based EEG models are promising because self-attention can capture long-range dependencies among feature tokens or temporal patches. This is particularly useful when stress patterns are distributed rather than localized.

## 2.4 ECG-Based Modeling

ECG reflects autonomic changes under stress through heart-rate and variability behavior. ECG-based stress systems commonly use statistical features with machine learning models such as logistic regression, random forest, and gradient boosting. Deep MLP models can perform well on fixed-length, high-dimensional ECG vectors with proper scaling and regularization.

## 2.5 Multimodal Fusion Approaches

Multimodal fusion methods are typically categorized as:

- **Early fusion:** concatenate raw features before modeling.  
- **Intermediate fusion:** encode each modality, then combine latent embeddings.  
- **Late fusion:** combine modality-specific predictions.

Intermediate fusion is generally preferred when modalities have different structures but meaningful interactions. Attention-based intermediate fusion can assign adaptive importance to cross-modal signals.

## 2.6 Transformer Architectures in Biosignal Learning

Transformers have shown strong performance in sequence and tabular-token settings due to attention mechanisms. For biomedical tasks, attention improves context modeling and can capture long-range relationships that recurrent methods may miss.

Cross-attention enables one modality to condition on another modality, making it a strong choice for EEG-ECG integration where neural and autonomic mechanisms are correlated.

## 2.7 Class Imbalance and Evaluation

Stress datasets are often imbalanced. In such settings, accuracy alone can be misleading because majority classes dominate the metric. Recommended practices include:

- weighted training objectives,  
- Focal Loss or weighted cross-entropy,  
- macro-averaged metrics (macro F1),  
- balanced accuracy, and  
- per-class precision/recall/F1 reporting.

## 2.8 Research Gap

Despite progress, practical stress systems still face key gaps:

1. Limited interaction-aware multimodal fusion in many implementations.  
2. Inadequate imbalance-focused optimization and reporting.  
3. Lack of reproducible, deployment-oriented artifact workflows.

This project addresses all three by combining cross-attention fusion, class-aware training, and export-ready outputs.

---

# CHAPTER 3
# DATASET AND PREPROCESSING

## 3.1 Overview

This work uses EEG and ECG feature datasets aligned to a five-class stress prediction target. The project implementation is configuration-driven and includes preprocessing utilities for feature extraction, scaling, and class-weight computation.

## 3.2 Data Sources and Label Space

The stress-label space consists of:

- Class 0: Calm  
- Class 1: Mild Stress  
- Class 2: Moderate Stress  
- Class 3: High Stress  
- Class 4: Severe Stress

The implementation aligns labels into a common index range (0 to 4) to support unified training.

## 3.3 EEG Data Pipeline

The EEG branch uses 178 features per sample. In preprocessing:

1. EEG CSV data is loaded.  
2. Feature and label columns are extracted based on configuration.  
3. Features are validated for expected range behavior.  
4. Stratified splitting is used for train/validation partitioning.

The EEG encoder then treats the feature vector as a token sequence for Transformer processing.

## 3.4 ECG Data Pipeline

The ECG branch also uses 178 feature dimensions. Processing steps include:

1. Load ECG CSV with metadata columns.  
2. Extract numerical feature columns and label column.  
3. Align class labels to 0 to 4 index space.  
4. Normalize features with `StandardScaler` fit on training data only.  
5. Reuse fitted scaler for validation/test data to avoid leakage.

## 3.5 Label Alignment and Pairing

Multimodal training requires consistent pairing between EEG features, ECG features, and labels. The implementation uses aligned indexing and a shared class mapping. This design allows synchronized sample feeding into the fusion model.

## 3.6 Train/Validation/Test Strategy

The training process includes:

- stratified train/validation split to preserve class proportions,  
- held-out test evaluation after model selection,  
- early stopping using validation macro F1.

This prevents direct test-set optimization and supports reliable performance estimation.

## 3.7 Class Imbalance Profile

Observed class distribution indicates nonuniform representation across stress levels. Minority classes are significantly smaller than Calm and Severe classes in available data. To mitigate imbalance effects:

- class weights are computed from training labels, and  
- Focal Loss is used to emphasize harder examples.

## 3.8 Preprocessing Summary

Final preprocessing outputs include:

- normalized EEG/ECG feature tensors,  
- aligned labels,  
- train/validation/test dataloaders,  
- class weight vector, and  
- serialized scalers for inference consistency.

---

# CHAPTER 4
# METHODOLOGY

## 4.1 Overview

NeuroFusionGPT is a four-stage model:

1. EEG Transformer Encoder  
2. ECG MLP Encoder  
3. Cross-Attention Fusion  
4. Stress Classifier

## 4.2 Problem Formulation

For each sample \(i\), let:

- \(x_i^{eeg} \in \mathbb{R}^{178}\)  
- \(x_i^{ecg} \in \mathbb{R}^{178}\)  
- \(y_i \in \{0,1,2,3,4\}\)

The model \(f_\theta\) predicts:

\[
\hat{y}_i = \arg\max f_\theta(x_i^{eeg}, x_i^{ecg})
\]

with softmax probability output for all five classes.

## 4.3 EEG Transformer Encoder

The EEG encoder projects each scalar token into a latent dimension \(d=128\), adds sinusoidal positional encoding, and passes the sequence through Transformer encoder layers (multi-head self-attention + feed-forward blocks).

Configuration used:

- \(d_{model} = 128\)  
- heads = 8  
- layers = 4  
- dropout = 0.1  
- pooling = mean

The pooled output is layer-normalized to obtain a fixed EEG embedding.

## 4.4 ECG MLP Encoder

The ECG encoder is a feed-forward network:

\[
178 \rightarrow 256 \rightarrow 128
\]

with BatchNorm, ReLU, and dropout. This produces a compact ECG embedding in the same latent space as EEG.

## 4.5 Cross-Attention Fusion

Given embeddings \(E\) and \(C\), cross-attention is applied bidirectionally:

\[
E' = \text{Norm}(E + \text{MHA}(Q=E, K=C, V=C))
\]
\[
C' = \text{Norm}(C + \text{MHA}(Q=C, K=E, V=E))
\]

Then:

\[
F = W_f [E' || C']
\]

where \(F \in \mathbb{R}^{128}\) is the fused representation used by the classifier.

This design explicitly models inter-modal relevance rather than assuming independent modalities.

## 4.6 Classification Head

The classifier maps fused embeddings to stress logits:

\[
128 \rightarrow 256 \rightarrow 5
\]

with ReLU and dropout in the hidden stage.

## 4.7 Loss Function and Optimization

To address class imbalance, Focal Loss is used:

\[
\mathcal{L}_{focal} = -\alpha_t (1-p_t)^\gamma \log(p_t)
\]

where:

- \(p_t\) is probability of the true class,  
- \(\alpha_t\) is class weight,  
- \(\gamma = 2.0\) controls focus on hard examples.

Optimization settings:

- Optimizer: AdamW  
- Learning rate: 0.0003  
- Weight decay: 0.01  
- Scheduler: ReduceLROnPlateau (monitor validation macro F1)

## 4.8 Regularization and Early Stopping

Generalization controls include:

- dropout in encoders and classifier,  
- gradient clipping (max norm = 1.0),  
- validation-based early stopping (patience = 10),  
- best-checkpoint selection by validation macro F1.

## 4.9 Evaluation Metrics

Primary metrics:

- Macro F1-score  
- Balanced Accuracy

Secondary metrics:

- Accuracy  
- Weighted F1-score  
- Per-class precision, recall, F1  
- Confusion matrix and ROC curves

These metrics provide both global and minority-class sensitive evaluation.

---

# CHAPTER 5
# ORIGINAL NEW IMPLEMENTATIONS

## 5.1 Overview

This chapter presents the implementation contributions beyond a standard baseline pipeline.

## 5.2 NeuroFusionGPT Architecture Implementation

The implementation defines modular components for EEG encoding, ECG encoding, fusion, and final classification. The cross-attention module is implemented with multi-head attention, residual connections, and normalization to preserve stable gradients and interaction-aware learning.

A dedicated `NeuroFusionGPT` class combines:

- `EEGEncoder`  
- `ECGEncoder`  
- `CrossAttentionFusion`  
- classifier head

This modular structure makes experimentation (e.g., ablations) straightforward.

## 5.3 Imbalance-Aware Training Pipeline

The trainer class includes:

- automatic class-weight integration,  
- Focal Loss instantiation from configuration,  
- learning-rate scheduling,  
- epoch-level metric tracking,  
- early stopping by macro F1,  
- best/last checkpoint export.

This setup explicitly prioritizes class-balanced model quality.

## 5.4 Experiment Tracking and Artifact Export

The project exports a full artifact set:

- `best_fusion_model.pth`  
- `scaler_eeg.pkl` and `scaler_ecg.pkl`  
- `model_config.json`  
- `fusion_results.json`  
- `training_history.json`

This supports reproducibility, auditability, and direct downstream deployment.

## 5.5 Visualization and Error Analysis Pipeline

The notebook and scripts generate:

- data EDA views,  
- training curves (loss, accuracy, F1),  
- confusion matrices (raw and normalized),  
- per-class metric charts,  
- ROC curves by class,  
- confidence analysis plots.

These visual outputs are essential for interpretation and thesis reporting.

## 5.6 Optional LLM Feedback Integration

The inference layer includes OpenRouter API integration for generating short wellness guidance from predicted stress label, confidence, and class probabilities. This creates a human-readable endpoint for practical application without altering classifier behavior.

---

# CHAPTER 6
# RESULTS AND DISCUSSION

## 6.1 Overview

Training was configured for up to 50 epochs with early stopping. The model reached its best validation macro F1 at epoch 27 and stopped at epoch 37 due to no further improvement.

## 6.2 Training Dynamics

Training logs indicate steady convergence with gradual validation improvement. The scheduler reduced learning rate when validation macro F1 plateaued. Early stopping prevented late overfitting and preserved the best-performing checkpoint.

Key observations:

- strong early gains in validation quality,  
- stable balanced accuracy through mid-to-late epochs,  
- diminishing returns after best validation epoch.

## 6.3 Quantitative Results

### Table 1. NeuroFusionGPT Training Configuration

| Component | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 0.0003 |
| Weight Decay | 0.01 |
| Batch Size | 128 |
| Maximum Epochs | 50 |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | Patience = 10 (macro F1 monitor) |
| Loss | Focal Loss (gamma = 2.0) + class weights |
| Best Validation Macro F1 | 0.8010 |
| Best Epoch | 27 |
| Stop Epoch | 37 |

### Table 2. Test-set Overall Metrics

| Metric | Value |
|---|---:|
| Loss | 0.2053 |
| Accuracy | 0.9315 |
| Balanced Accuracy | 0.9133 |
| Macro F1-score | 0.7826 |

### Table 3. Per-Class Performance Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Calm | 0.9933 | 0.9289 | 0.9600 | 18118 |
| Mild Stress | 0.3621 | 0.8147 | 0.5014 | 556 |
| Moderate Stress | 0.8206 | 0.9537 | 0.8821 | 1448 |
| High Stress | 0.4704 | 0.8827 | 0.6137 | 162 |
| Severe Stress | 0.9269 | 0.9863 | 0.9557 | 1608 |

## 6.4 Class-wise Analysis

The model performs strongly on **Calm** and **Severe Stress**, reflecting robust decision boundaries for well-represented classes. For minority classes (**Mild Stress**, **High Stress**), recall is high but precision is lower. This indicates that the classifier successfully detects many minority instances but also introduces false positives when class boundaries overlap.

This behavior is common in imbalance-aware systems where the optimization objective intentionally increases sensitivity to rare classes.

## 6.5 Discussion of Strengths and Weaknesses

### Strengths

1. Strong global and balanced metrics on held-out data.  
2. Interaction-aware multimodal fusion via cross-attention.  
3. Reproducible and deployment-friendly artifact outputs.  
4. Class-sensitive analysis beyond overall accuracy.

### Weaknesses

1. Minority precision still needs improvement.  
2. Pairing strategy may not fully guarantee subject/time synchronization.  
3. Domain shift risk if evaluated on different sensor setups or populations.  
4. Attention interpretability is not yet fully quantified with attribution studies.

## 6.6 Threats to Validity

### Table 5. Risk and Limitation Summary

| Risk | Impact | Current Mitigation | Future Mitigation |
|---|---|---|---|
| Class imbalance | Inflated majority performance | Focal Loss + class weights | Targeted augmentation and calibration |
| Pairing mismatch | Noisy multimodal correspondence | Index-based alignment checks | Subject-level synchronization |
| Dataset bias | Reduced generalization | Stratified splitting | Cross-dataset evaluation |
| Overfitting | Degraded test behavior | Early stopping + dropout | Stronger regularization and ensembling |

## 6.7 Suggested Ablation and Extension Experiments

### Table 4. Suggested Ablation Template

| Model Variant | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| EEG-only | [fill] | [fill] | [fill] |
| ECG-only | [fill] | [fill] | [fill] |
| Concat fusion | [fill] | [fill] | [fill] |
| Cross-attention fusion (proposed) | 0.9315 | 0.9133 | 0.7826 |

Recommended next experiments:

1. Subject-wise split validation.  
2. Calibration analysis (ECE/Brier score).  
3. Uncertainty-aware inference with thresholding.  
4. Missing-modality robustness testing.  
5. External dataset transfer evaluation.

---

# CHAPTER 7
# CONCLUSION AND FUTURE WORK

## 7.1 Conclusion

This project developed and evaluated **NeuroFusionGPT**, a multimodal stress classification framework combining EEG and ECG through cross-attention fusion. The model achieved strong held-out performance:

- Accuracy: 0.9315  
- Balanced Accuracy: 0.9133  
- Macro F1: 0.7826

The architecture effectively leverages complementary neural and cardiac information. The training setup addressed class imbalance with Focal Loss and class weights, while early stopping and scheduler control supported stable convergence.

Beyond predictive performance, the project contributes a practical workflow with reproducible artifacts, evaluation visualizations, and optional natural-language feedback integration for end-user applications.

## 7.2 Future Work

Future improvements include:

1. Collect synchronized, same-subject multimodal stress datasets.  
2. Improve minority precision using augmentation and calibration.  
3. Perform complete ablation comparisons and significance testing.  
4. Add uncertainty-aware decision logic for safer deployment.  
5. Investigate edge/mobile inference optimization.  
6. Strengthen privacy, consent, and ethical controls for biosignal applications.

---

# REFERENCES

[1] A. Vaswani et al., "Attention Is All You Need," Advances in Neural Information Processing Systems, 2017.  
[2] T.-Y. Lin et al., "Focal Loss for Dense Object Detection," ICCV, 2017.  
[3] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.  
[4] B. Shickel, P. Tighe, A. Bihorac, and P. Rashidi, "Deep EHR: A Survey of Recent Advances in Deep Learning Techniques for Electronic Health Record Analysis," IEEE Journal of Biomedical and Health Informatics, 2018.  
[5] H. Song, D. Rajan, J. Thiagarajan, and A. Spanias, "Attend and Diagnose: Clinical Time Series Analysis using Attention Models," AAAI, 2018.  
[6] E. Alsentzer et al., "Publicly Available Clinical BERT Embeddings," arXiv:1904.03323, 2019.  
[7] J. Lee et al., "BioBERT: a pre-trained biomedical language representation model for biomedical text mining," Bioinformatics, 2020.  
[8] M. Ghassemi et al., "A Review of Challenges and Opportunities in Machine Learning for Health," AMIA Summits on Translational Science Proceedings, 2020.  
[9] S. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS, 2017.  
[10] C. Suresh et al., "Artificial Intelligence in the Intensive Care Unit: Current Evidence on an Inevitable Future Tool," Cureus, 2024.  
[11] A. Johnson et al., "MIMIC-III, a freely accessible critical care database," Scientific Data, 2016.  
[12] X. Zhang et al., "A Review on Human Stress Detection from Physiological Signals," IEEE Access, [add volume/year/pages].  
[13] [Add at least 5 domain-specific EEG/ECG stress papers required by your advisor.]  

---

## APPENDIX A - VIVA QUESTIONS AND SHORT ANSWERS

**Q1. Why EEG + ECG instead of one modality?**  
Because stress affects both neural and autonomic systems; multimodal fusion captures complementary evidence and improves robustness.

**Q2. Why cross-attention instead of concatenation?**  
Cross-attention explicitly models interactions between modalities. Concatenation assumes independent features and may miss conditional relationships.

**Q3. Why macro F1 and balanced accuracy?**  
The data is imbalanced; these metrics better reflect minority-class performance than accuracy alone.

**Q4. How do you prevent data leakage?**  
Scalers are fit only on training data and reused for validation/test. Model selection is based on validation, not test.

**Q5. What are the major limitations?**  
Minority precision, pairing assumptions, and external generalization remain open challenges.

**Q6. Where is "GPT" in NeuroFusionGPT?**  
The classifier is not a generative language model. "GPT" refers to optional application-layer LLM integration for natural-language feedback.

---

## APPENDIX B - FINAL SUBMISSION CHECKLIST

- [ ] Replace all placeholders (name, advisor, department, institution, semester).  
- [ ] Add required university certificate/declaration pages.  
- [ ] Insert all figures and finalize figure numbering.  
- [ ] Update page numbers in ToC/List of Tables/List of Figures.  
- [ ] Add advisor-approved references and citation style (IEEE/APA).  
- [ ] Run grammar and plagiarism checks per department policy.  
- [ ] Export final document to DOCX and PDF in university format.  

