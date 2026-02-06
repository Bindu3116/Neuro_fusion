# Questions 


---

## 1. **Why fuse EEG and ECG for stress? What does each modality add?**

ECG gives direct autonomic stress markers (heart rate, HRV) and we use its labels. EEG adds neurological context; cross-attention learns how brain and heart patterns relate. Brain–heart coupling can improve prediction and interpretability.

**If asked “But your EEG is epilepsy, not stress?”**  
We use ECG stress as the target. EEG is an auxiliary modality to enrich the representation; with stress-labeled EEG later, both can predict the same target.

---

## 2. **How does cross-attention fusion work? Why not just concatenate?**

Two 128‑dim embeddings; cross-attention is bidirectional: EEG attends to ECG and ECG to EEG. Outputs are normalized, concatenated, and projected to one 128‑dim vector. Cross-attention models interactions explicitly; concatenation treats modalities as independent.

---

## 3. **How do you align EEG (16/178) and ECG (187) features?**

Each modality has its own encoder; alignment is in embedding space. EEG → Transformer → 128‑dim; ECG → MLP → 128‑dim. Standard scaling is fit on train only and reused at inference.

---

## 4. **How do you handle class imbalance? Why macro F1?**

Focal loss and class weights (inverse frequency). Early stopping and best-model selection on macro F1 so we optimize per-class performance, not just accuracy.

---

## 5. **Train/validation/test splits and data leakage?**

Fixed ratio (e.g. 90% train, 10% val); test held out for final metrics. Scalers fit only on training data. Same split logic for both modalities when aligning by index.

---

## 6. **How do you know the model uses both modalities?**

Ablation (EEG-only, ECG-only, both); attention visualization; corrupt one modality (e.g. zero EEG) and check if performance drops.

---

## 7. **What is “NeuroFusionGPT”? Where is the GPT?**

Core model = EEG transformer + ECG MLP + cross-attention (stress prediction). “GPT” = optional LLM in the app layer that takes the prediction and generates wellness advice.

---

## 8. **Why Transformer for EEG and MLP for ECG?**

EEG has sequence structure (channels/time); Transformer captures those dependencies. ECG is a fixed-length feature vector; MLP is a good fit. Different inductive biases, then fuse in shared 128‑dim space.

---

## 9. **Limitations and threats to validity?**

Label mismatch (EEG from epilepsy, target from ECG stress). Pairing by index may not be same person or time. Single datasets → generalization and interpretability need more validation.

---

## 10. **Future work?**

Stress-labeled EEG; ablations and attention viz; robustness (splits, cross-dataset); deployment (real-time, on-device, privacy).

---

## 11. **Why 128-dimensional embeddings? Why 5 classes?**

128 is a balance between capacity and overfitting; common in transformer mid-layers. 5 classes match the stress dataset labels (e.g. Calm → Severe). Both are set by the data and standard practice.

---

## 12. **What is positional encoding for in the EEG encoder?**

EEG is treated as a sequence (channels or time). Positional encoding gives the Transformer order information so it knows which channel or time step is which.

---

## 13. **What optimizer and learning rate? How do you prevent overfitting?**

AdamW with a small learning rate (e.g. 1e–4 to 3e–4) and weight decay. Overfitting: dropout in encoders and classifier, early stopping on validation metric, optional learning-rate schedule.

---

## 14. **What metrics do you report besides macro F1?**

Accuracy, balanced accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrix, and optionally ROC curves for each class.

---

## 15. **What is focal loss in one line?**

Loss that down-weights easy examples (high confidence) so the model focuses more on hard or rare classes; helps with imbalance.

---

## 16. **Why multi-head attention? How many heads?**

Heads learn different types of relationships (e.g. different channel pairs). We use a small number (e.g. 4–8) so the model stays tractable; d_model must be divisible by number of heads.

---

## 17. **What if one modality is missing at test time?**

Current design expects both. For missing modality: could train with dropout on one branch, or add a separate “EEG-only / ECG-only” path and route at inference.

---

## 18. **How would you deploy this in practice?**

Export the fusion model; run inference on device or server. Preprocess with the same scalers. Optionally call an LLM API with the predicted stress level to return text advice. Consider latency, privacy, and consent.

---

## 19. **Real-time vs batch?**

Trained on batches. For real-time: stream short windows of EEG/ECG, run inference per window; latency depends on window length and hardware.

---

## 20. **How did you choose hyperparameters?**

Started from common defaults (e.g. d_model=128, 4–8 heads, dropout 0.1–0.3). Tuned learning rate and early-stopping patience on validation performance; can use grid search or manual sweeps.

---

## 21. **Comparison with existing stress-detection methods?**

We add multimodal fusion (EEG+ECG) and cross-attention. Baseline comparison: ECG-only or EEG-only with same backbone; report macro F1 / accuracy to show gain from fusion.

---

## 22. **What about ethical considerations?**

Informed consent for biosignals; clear explanation of stress prediction and LLM advice; data and model privacy; avoid overclaiming clinical validity without proper validation.

---

## 23. **How long does training take? What hardware?**

Depends on data size and hardware. On a single GPU (e.g. Colab), tens of epochs can take on the order of minutes to an hour. We train in Colab with optional GPU.

---

## 24. **What activation functions?**

ReLU in MLPs; GELU in the Transformer; softmax in attention and in the final classifier for class probabilities.

---

## 25. **Why save the “best” model by validation metric?**

Early stopping avoids overfitting; saving the best by validation macro F1 (or similar) gives the model that generalizes best, not the last epoch.

---
