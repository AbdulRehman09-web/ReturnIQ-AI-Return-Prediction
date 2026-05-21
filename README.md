# ReturnIQ — AI-Powered E-Commerce Return Prediction & Revenue Loss Estimation

<div align="center">

## 📦 Hybrid AI Intelligence Platform for E-Commerce Analytics

### Predict Product Returns • Estimate Revenue Loss • Explain AI Decisions

</div>

---

# 📌 Project Overview

ReturnIQ is an advanced Hybrid AI system designed to predict e-commerce product return risk and estimate potential revenue loss using Machine Learning, Deep Learning, NLP, Explainable AI, and Business Rule Intelligence.

The system combines:

* Machine Learning Models
* Deep Learning Models
* Natural Language Processing (NLP)
* Explainable AI (SHAP)
* Rule-Based Decision Intelligence
* Interactive Business Dashboards
* API-based Deployment Architecture

ReturnIQ analyzes:

* Customer reviews
* Payment behavior
* Freight cost
* Product information
* Transaction patterns

to generate:

* Return probability
* Revenue loss estimation
* Risk factors
* Business recommendations
* Explainable insights

---

# 🎯 Problem Statement

E-commerce businesses face significant losses due to:

* Product returns
* Fraudulent transactions
* Negative customer experiences
* Delivery issues
* High shipping costs

Traditional systems only react after returns occur.

ReturnIQ provides a proactive AI-driven solution that predicts return risk before business loss happens.

---

# 🚀 Key Features

## ✅ AI-Powered Return Prediction

Predicts whether an order is likely to be returned.

## ✅ Revenue Loss Estimation

Estimates financial risk associated with returned orders.

## ✅ Explainable AI

Uses SHAP explainability and rule-based explanations to interpret AI decisions.

## ✅ NLP-Based Review Analysis

Analyzes customer review text using TF-IDF and Deep Learning.

## ✅ Hybrid Decision Architecture

Combines:

* XGBoost ML predictions
* Rule-based intelligence
* NLP signals

## ✅ Interactive Dashboard

Luxury enterprise-style dashboard built with Streamlit.

## ✅ FastAPI Backend

Production-style inference API with diagnostics endpoints.

## ✅ Multiple AI Models

Includes:

* Decision Tree
* Random Forest
* XGBoost
* DNN
* LSTM
* CNN

## ✅ Business Recommendations Engine

Generates actionable recommendations based on prediction signals.

---

# 🧠 Technologies Used

## Programming Language

* Python

## Machine Learning

* Scikit-learn
* XGBoost
* Random Forest
* Decision Tree

## Deep Learning

* TensorFlow
* Keras
* DNN
* LSTM
* 1D CNN

## NLP

* TF-IDF Vectorization
* Tokenization
* Sequence Padding

## Explainable AI

* SHAP

## Backend

* FastAPI

## Frontend

* Streamlit

## Data Processing

* Pandas
* NumPy
* SciPy Sparse Matrices

## Visualization

* Matplotlib
* Seaborn

---

# 📂 Dataset

This project uses the Brazilian E-Commerce Public Dataset by Olist.

Merged datasets include:

* Orders
* Order Items
* Payments
* Reviews
* Customers
* Products

---

# 🏗️ System Architecture

```text
                ┌─────────────────────┐
                │  Streamlit Frontend │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     FastAPI API     │
                └──────────┬──────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌────────────────┐  ┌────────────────┐
│ XGBoost ML   │  │ Rule Engine    │  │ TF-IDF NLP     │
│ Classifier   │  │ Intelligence   │  │ Review Analysis│
└──────────────┘  └────────────────┘  └────────────────┘
        │                  │
        └──────────┬───────┘
                   ▼
        ┌────────────────────┐
        │ Hybrid AI Decision │
        └──────────┬─────────┘
                   ▼
        ┌────────────────────┐
        │ Random Forest Loss │
        │ Estimation Model   │
        └────────────────────┘
```

---

# 📊 Machine Learning Pipeline

## 1. Data Collection & Merging

Multiple Olist datasets are merged into one unified dataframe.

## 2. Data Cleaning

* Missing value handling
* Invalid date handling
* Numeric conversion
* Duplicate checks

## 3. Feature Engineering

Features include:

* payment_value
* price
* freight_value
* delivery_time
* review_score
* review_comment_message

## 4. NLP Processing

Customer reviews are converted into numerical vectors using TF-IDF.

```python
TfidfVectorizer(max_features=300)
```

## 5. Feature Matrix

Final feature matrix:

* 3 numeric features
* 300 TF-IDF text features
* Total = 303 features

## 6. SMOTE Balancing

SMOTE is used to handle class imbalance.

## 7. Feature Scaling

StandardScaler is applied using sparse matrix support.

---

# 🤖 Classification Models

| Model         | Purpose                        |
| ------------- | ------------------------------ |
| Decision Tree | Baseline classification        |
| Random Forest | Ensemble learning              |
| XGBoost       | Best performing classifier     |
| DNN           | Deep feature learning          |
| LSTM          | Sequential text understanding  |
| 1D CNN        | Local phrase pattern detection |

---

# 📈 Regression Models

Used for revenue loss estimation.

| Model                   | Purpose              |
| ----------------------- | -------------------- |
| Linear Regression       | Baseline regression  |
| Decision Tree Regressor | Nonlinear regression |
| Random Forest Regressor | Best regressor       |
| XGBoost Regressor       | Boosted regression   |
| Neural Network          | Deep regression      |

---

# 🧠 Deep Learning Models

## DNN

Fully connected neural network:

```text
256 → 128 → 64 → 1
```

## LSTM

Used for sequential text analysis:

* Embedding Layer
* LSTM Layer
* Dense Output Layer

## 1D CNN

Used for phrase-level sentiment pattern extraction:

* Conv1D
* GlobalMaxPooling
* Dense Layers

---

# 🔍 Explainable AI (SHAP)

SHAP is used to:

* Interpret model predictions
* Understand feature importance
* Visualize AI decisions

Generated explainability:

* SHAP Summary Plot
* SHAP Feature Importance Bar Plot

---

# ⚡ Hybrid AI Decision System

ReturnIQ uses a Hybrid AI architecture.

## ML Prediction

XGBoost generates probability scores.

## Rule-Based Intelligence

Additional business rules evaluate:

* Low review score
* Heavy products
* Freight ratio
* Payment anomalies
* Long installment plans
* Negative review keywords

## Final Decision

Hybrid decision combines:

* ML probability
* Rule score
* Business intelligence

---

# 📉 Blended Probability System

The project uses a blended probability mechanism:

```python
blended_probability =
    0.55 * ml_probability
  + 0.45 * rule_signal
```

This ensures:

* Better interpretability
* Responsive UI predictions
* Stronger business logic integration

---

# 📊 Visualizations Generated

The project generates:

* ROC Curves
* Confusion Matrices
* Accuracy Comparison Charts
* Regression Comparison Charts
* SHAP Plots
* DNN Loss Curves
* DNN Accuracy Curves
* LSTM ROC Curves
* Prediction Distributions
* Correlation Heatmaps
* Word Clouds

---

# 🖥️ Streamlit Dashboard

The frontend dashboard includes:

* Interactive sidebar inputs
* Return probability visualization
* Revenue risk metrics
* Explainability cards
* Risk factor analysis
* Recommendation engine
* Debug trace panel
* Model performance tabs

---

# 🔌 FastAPI Backend

FastAPI handles:

* Prediction inference
* Feature engineering
* Rule scoring
* Probability blending
* Explanation generation

## API Endpoints

| Endpoint         | Purpose                  |
| ---------------- | ------------------------ |
| `/predict`       | Main prediction endpoint |
| `/health`        | Health monitoring        |
| `/debug/imports` | Diagnostics              |
| `/`              | API home                 |

---

# 📁 Project Structure

```text
ReturnIQ/
│
├── app/
│   └── app.py
│
├── api/
│   └── main.py
│
├── data/
│   └── raw/
│
├── models/
│   ├── scaler.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoders.pkl
│   ├── best_classification_model_XGBoost.pkl
│   └── best_model_RandomForest.pkl
│
├── notebooks/
│   ├── graphs/
│   └── results/
│
├── utils/
│   └── explain.py
│
├── requirements.txt
└── README.md
```

---

# ▶️ How to Run the Project

# 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/ReturnIQ-AI-Return-Prediction.git
cd ReturnIQ-AI-Return-Prediction
```

---

# 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 3️⃣ Run FastAPI Backend

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

---

# 4️⃣ Run Streamlit Frontend

```bash
streamlit run app/app.py
```

---

# 📊 Model Performance

## Best Classifier

✅ XGBoost Classifier

## Best Regressor

✅ Random Forest Regressor

---

# 📌 Future Improvements

Possible future enhancements:

* Real-time cloud deployment
* Docker containerization
* User authentication
* Database integration
* Real-time order streaming
* Transformer-based NLP models
* Advanced ensemble learning
* Recommendation personalization
* Fraud detection module
* CI/CD pipeline integration

---

# 🎓 Academic Scope

This project covers concepts from:

* Machine Learning
* Deep Learning
* Natural Language Processing
* Explainable AI
* Business Intelligence
* Full Stack AI Development
* API Engineering
* MLOps Foundations

---

# 👨‍💻 Contributors

* Arzoo Sarwari
* Talha Ahmed Khan
* Abdul Rehman

---

# 📜 License

This project is developed for educational and research purposes.

---

# ⭐ Acknowledgements

Special thanks to:

* Olist Dataset
* Scikit-learn
* TensorFlow
* XGBoost
* SHAP
* FastAPI
* Streamlit

