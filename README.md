# Predicting Recidivism: Neural Networks vs Logistic Regression

This repository contains the code, results, and extended essay for my IB Computer Science Extended Essay:

**Research Question:**
*"How far will a neural network model be more accurate in predicting recidivism than the logistic regression model used by COMPAS?"*

---

## 📘 About
Recidivism prediction is a critical task in criminal justice, used to assess the likelihood of a defendant re-offending. This project evaluates the performance of Neural Networks (NNs) compared to the Logistic Regression (LR) model employed by the COMPAS system, using a subset of the ProPublica Broward County dataset.

The aim is to determine whether a Neural Network can offer measurable improvements over traditional Logistic Regression when predicting future criminal behavior.

---

## 📂 Repository Structure

```
predicting-recidivism-nn-vs-lr/
├── README.md
├── extended-essay.pdf
├── data/
│   └── compas_dataset_sample.csv (placeholder or simulated data)
├── notebooks/
│   └── compas_logistic_regression.ipynb
│   └── compas_neural_network.ipynb
│   └── rfe_feature_selection.ipynb
├── src/
│   └── logistic_regression.py
│   └── neural_network.py
│   └── feature_selection.py
├── results/
│   └── evaluation_metrics.csv
│   └── model_comparison.png
├── LICENSE
└── requirements.txt
```

---

## 💻 How to Run

1. Clone this repository:
```bash
git clone https://github.com/yourusername/predicting-recidivism-nn-vs-lr.git
cd predicting-recidivism-nn-vs-lr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run experiments:
- Open Jupyter Lab or Notebook.
- Explore the code in the `/notebooks` folder to train models and reproduce results.

---

## 📊 Key Findings
- Neural Networks improved F1 scores by approximately **2.8%** compared to Logistic Regression models.
- Logistic Regression performed competitively when optimized with feature selection (RFE).
- NN models, although more computationally expensive, can detect more nuanced patterns in the data.

---

## ⚖️ License
MIT License — feel free to use, modify, and distribute with proper citation.

---

## 📚 References
A full list of references is included in the `extended-essay.pdf`.

---

Thanks for exploring this research — feedback and suggestions are welcome!


License

This project is for research purposes. If you use any part of this code, please provide attribution.

