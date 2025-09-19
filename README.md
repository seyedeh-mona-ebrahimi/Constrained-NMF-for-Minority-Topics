# 🟢 Constrained Non-negative Matrix Factorization for Guided Minority Topic Modeling

*"Discover minority topics in text datasets through guided, constraint-based topic modeling."*
---

### ⚠️ Review-Only Notice

This repository is provided **solely for peer review purposes**.

- The code is **not licensed for commercial or public use**.
- Redistribution, reuse, or publication of the code or data is **not permitted**.

If you're a reviewer, thank you for evaluating this work!

---

## 🧭 Overview

The **Constrained Non-negative Matrix Factorization** algorithm is a novel topic modeling framework designed to uncover **minority topics** in imbalanced text datasets.  
Traditional topic models tend to overlook low-frequency or underrepresented topics — our method integrates **seed word guidance** and **matrix constraints** to enhance minority topic discovery.

Key Highlights:
- 🎯 **Guided topic modeling** using user-defined seed words.
- 🔒 **Controlled constraints** over topic-word and document-topic matrices.
- 🟢 Effective discovery of **low-frequency, minority topics**.
- 🧪 Includes **evaluation scripts**

---

## 📂 Project Structure
```plaintext
📦 Minority_Topic-Model
│
├── data/                   → Synthetic datasets
├── Evaluation.py           → Evaluation script (NMI & Purity metrics across different methods)
├── script.py               → main script
├── script-run.py           → Parameter configuration script
├── sythtetic-data.csv      → Synthetic dataset
├── requirements.txt        → Python dependencies
└── README.md               → Project documentation
```







# 🚀 How to Run

## 🔧 Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

## 🧪 Step 2: Run the Main Training Script
To train the model using the synthetic dataset:
```bash
python script.py --input sythtetic-data.csv
```

## ⚙️ Step 3: Run with Custom Parameters
You can edit parameters inside script-run.py and then run it:
```bash
python script-run.py
```

## 📊 Step 4: Evaluate Model Performance
To compute NMI and Purity across baseline methods:
```bash
python Evaluation.py
```

***This will:
Compare model outputs to ground-truth labels and print or save metrics, also generate plots for evaluation.***

