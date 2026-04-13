# ML-Fake-News-Detection using Machine Learning

## Overview
This project focuses on detecting fake news articles using **Machine Learning**, specifically **Logistic Regression**.  
The model is trained on textual data and enhanced with **TF-IDF vectorization** and additional numerical features such as text length and word count.

---

##  Objectives
- Classify news articles as **Fake** or **Real**
- Apply **text preprocessing and feature engineering**
- Train and evaluate a **Logistic Regression model**
- Perform **cross-validation** and **data leakage checks**
- Generate visual insights and performance metrics

---

##  Project Structure:
Fake News Detection/
│
├── data/
│ └── WELFake_Dataset.csv (not included)
│
├── outputs/
│ ├── EDA visualizations
│ ├── Model evaluation plots
│ ├── Cross-validation results
│ ├── model.pkl
│ └── vectorizer.pkl
│
├── src/
│ └── Fake News Prediction.py
│
├── README.md
└── report.pdf



---

##  Dataset
- **Name:** WELFake Dataset  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification  

>  The dataset (~240MB) is not included in this repository due to size limitations.

---

##  Features Used
- **TF-IDF Features** (Top 3000 terms)
- **Text-based features:**
  - Text length
  - Word count
  - Title length

---

## Model Details
- **Algorithm:** Logistic Regression  
- **Solver:** saga  
- **Max Iterations:** 1000  
- **Train-Test Split:** 80:20  
- **Cross-Validation:** 5-Fold  

---

##  Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- ROC-AUC Score  
- Confusion Matrix  
- ROC Curve  

---

##  Additional Analysis
- Exploratory Data Analysis (EDA)
- Feature correlation heatmap
- Cross-validation performance comparison
- **Sanity Check using shuffled labels** (to detect data leakage)

---

##  Outputs
All outputs are saved in the `/outputs` folder:
- EDA graphs
- Model performance charts
- Confusion matrix
- ROC curve
- Cross-validation plots
- Serialized model and vectorizer (`.pkl` files)
---

