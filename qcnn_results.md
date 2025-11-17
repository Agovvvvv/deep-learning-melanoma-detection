# 🧬 QCNN for Melanoma Detection: A Research Recap

This document summarizes the series of experiments we conducted to determine if a hybrid **Quantum Convolutional Neural Network (QCNN)** could offer a performance advantage over a classical CNN for melanoma classification, especially in a low-data environment.

---

## 🎯 The Overall Goal

Our objective was to test the hypothesis that a QCNN, by leveraging the high-dimensional processing space of quantum mechanics, could learn more complex and generalizable patterns from a **small training set (3000 images)** than a purely classical model of equivalent size.

---

## 🧪 Experiment 1: The "Bottleneck" Model

Our first attempt used the `qcnn.ipynb` notebook.

### Architecture
- **Feature Extractor**: A very deep, pre-trained, and frozen **ConvNeXtBase** (87M parameters)
- **Classifier**: A small 8-qubit quantum circuit attached at the very end

### Result
❌ The quantum layer performed **no better** than a simple classical linear layer.

### 💡 Key Finding
The powerful classical model had already "solved" the problem. The quantum circuit was **"starved" of information**, as the rich 1024-dim features were crushed down to 8 dimensions. This was an **information bottleneck**.

---

## 🔬 Experiment 2: The "Quanvolutional" Model (The "True" QCNN)

Based on the failure of Experiment 1, we designed a new architecture to test the QCNN as a **core processing layer**, not an afterthought.

### New Architecture

1. **Shallow Classical CNN**: A small, 3-layer classical CNN (trained from scratch) to extract a low-level feature map (e.g., `[B, 8, 56, 56]`)
2. **Quantum Convolutional Layer**: A PQC (Parameterized Quantum Circuit) acting as a "quantum filter." It scanned the feature map, processing one pixel (8 channels) at a time
3. **Classical Head**: A simple Linear layer for the final classification

### The "Control" Model
To ensure a fair comparison, we built a **ClassicControlCNN** with the exact same architecture, but replacing the quantum layer with a simple classical `nn.Conv2d(kernel_size=1)`.

> 🔑 This design isolated the one variable we wanted to test: **the quantum filter vs. the classical filter**.

---

### Iteration 2.1: The 2-Layer QCNN (Shallow + Complex Circuit)

**Circuit**: `qml.StronglyEntanglingLayers(n_layers=2)`

**Result**: The QCNN and the ClassicControlCNN performed **identically**.

**Conclusion**: The 2-layer quantum filter was not powerful enough (or the task was not complex enough) to find any patterns that the simple 1×1 classical convolution couldn't also find.

---

### Iteration 2.2: The 4-Layer QCNN (Deeper + Complex Circuit)

**Hypothesis**: A deeper quantum circuit (4 layers) might be powerful enough to find an advantage.

**Result**: ❌ The QCNN performed **worse** than the classical model (AUC **0.8048** vs **0.8171**), as seen in `image_529d6a.png`.

**Conclusion**: The 4-layer `StronglyEntanglingLayers` circuit was **too complex**. It "overfit" to the `pos_weight` in the loss function, creating a biased model that was obsessed with "recall" at the expense of all other metrics.

---

### Iteration 2.3: The 16-Qubit QCNN (Wider Circuit)

**Hypothesis**: A wider circuit (16 qubits) would reduce the information bottleneck by allowing 16 channels.

**Result**: ⏱️ A single epoch was **"taking forever."**

**Conclusion**: The exponential scaling of quantum simulation (2¹⁶ vs. 2⁸) made this computationally infeasible on classical hardware.

---

## 🏆 Experiment 3: The Final, Definitive Test

We combined all our learnings into one final, robust experiment.

### Hypothesis
The 4-layer QCNN failed because:
- The circuit (`StronglyEntanglingLayers`) was **too complex** and prone to Barren Plateaus
- The loss function (`BCEWithLogitsLoss` + `pos_weight`) created a bad bias

### The "Best Shot" QCNN

- **Circuit**: `qml.BasicEntanglerLayers(n_layers=4)` — A simpler, more trainable circuit
- **Dataset**: A balanced 50/50 training set (3000 images)
- **Loss**: A standard `nn.BCEWithLogitsLoss` (no `pos_weight` needed)

### The Test
This "best shot" QCNN and its identical ClassicControlCNN were trained on the **3000 balanced images** and then evaluated on a massive, unseen **9,942-image test set** (`image_a4d1e2.png`).

---

## 📊 Final Result

The results from `image_a4d1e2.png` were **definitive**.

| Metric | QCNN (4-Layer BasicEntangler) | Classic Control (1×1 Conv) | 🏅 Winner |
|--------|-------------------------------|----------------------------|-----------|
| **AUC** | 0.8307 | **0.8398** | 🟦 Classic |
| **Accuracy** | 0.7138 | **0.7341** | 🟦 Classic |
| **F1-Score** | 0.5513 | **0.5675** | 🟦 Classic |
| **Recall** | **0.7903** | 0.7844 | 🟨 QCNN |
| **Precision** | 0.4233 | **0.4445** | 🟦 Classic |

### Analysis
The QCNN learned to be a **"recall-focused"** model, but in doing so, it sacrificed precision and overall accuracy. The classical model was superior on every key metric (**AUC, Accuracy, F1-Score**), proving to be a more balanced and effective classifier.

---

## 🎓 Overall Research Conclusion

Our extensive, iterative experiments consistently found that for this image classification task, the **classical CNN was the superior model**. The QCNN, in all tested configurations, either performed identically or slightly worse than its classical counterpart, all while being **exponentially slower to train**. 

> ⚠️ This suggests that this specific architecture and task are **not well-suited for a "quantum advantage"**.