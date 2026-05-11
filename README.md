# 🏨 Hotel Booking Cancellation Risk Tool

**Predicting hotel booking cancellations to support revenue management decisions**

Built by Noah Johnson

---

## 🚀 Live App
**[hotel-cancel-predictor.streamlit.app](https://hotel-cancel-predictor.streamlit.app)**

---

## 📌 Project Overview

Hotel cancellations represent a significant revenue risk for the hospitality 
industry. This project builds a machine learning system that predicts whether 
a hotel booking will be cancelled before arrival, and uses a large language 
model (Claude by Anthropic) to translate those predictions into plain-English 
recommendations a revenue manager can act on.

**Target user:** Revenue Managers at mid-to-large hotel chains (Marriott, 
Hilton, independent groups)

**Core question:** Can a model identify individual high-risk bookings more 
accurately than simple rule-of-thumb heuristics, at a scale no manual 
process can match?

---

## 🤖 What the App Does

- **Cancellation Risk Score** — Enter booking details and get an XGBoost 
  model prediction (ROC-AUC: 0.92)
- **AI Recommendation** — Claude (Anthropic API) generates a plain-English 
  risk summary and recommended action for the revenue manager
- **Natural Language Search** — Ask questions about bookings in plain English; 
  Claude converts them to data filters automatically

---

## 📊 Model Performance

| Model | Accuracy | F1 (cancellations) | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline) | 0.77 | 0.45 | 0.77 |
| Random Forest | 0.85 | 0.71 | 0.91 |
| **XGBoost (final)** | **0.85** | **0.72** | **0.92** |

---


## 🗂️ Project Structure

hotel-cancellation-predictor/
app.py                        ← Streamlit app (final product)
xgb_model.pkl                 ← Trained XGBoost model
feature_columns.pkl           ← Feature column order for inference
requirements.txt              ← Python dependencies
.python-version               ← Python 3.11 for Streamlit Cloud
notebooks/
hotel_cancellation_project.ipynb  ← Full project notebook (Colab)
data/
README.md                   ← Dataset download instructions
src/                          ← Reusable code (future work)
reports/                      ← Figures and outputs

---

## ⚙️ Setup & Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/johnsonnoah2000/hotel-cancellation-predictor
cd hotel-cancellation-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `hotel_bookings.csv` from Kaggle and place it in the `data/` folder:
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

### 4. Run the notebook
Open `notebooks/hotel_cancellation_project.ipynb` in Google Colab.
Add your Anthropic API key to Colab Secrets as `ANTHROPIC_API_KEY`.

### 5. Run the Streamlit app
Create a `.streamlit/secrets.toml` file:
```toml
ANTHROPIC_API_KEY = "your-key-here"
```
Then run:
```bash
streamlit run app.py
```

---

## 📦 Dependencies
pandas, numpy, matplotlib, seaborn
scikit-learn, xgboost, shap
anthropic, streamlit, joblib

---

## ⚠️ Limitations

- Dataset covers two hotels in Portugal (2015–2017) — predates COVID-19
- Model may not generalize to other regions or current booking behavior 
  without retraining
- Production deployment would require direct PMS integration (Oracle Opera, 
  Amadeus, Salesforce Hospitality) to auto-score bookings without manual input

---

## 📚 Data Source

Antonio, N., Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. 
*Data in Brief*, 22, 41–49.
Kaggle: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
License: CC0 1.0 Public Domain

---

## 🤝 AI Tools Used

This project used Claude (Anthropic) for code generation, project planning, 
and as the LLM layer in the final product. All outputs were verified and 
modified by the author. Full AI usage log is documented in the notebook.
