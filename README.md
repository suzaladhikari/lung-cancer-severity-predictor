# 🎗️ Lung Cancer Severity Predictor

A **Streamlit-powered machine learning web app** that predicts the **severity level of lung cancer** — Low, Medium, or High — based on lifestyle and clinical input features.

> ⚠️ **Disclaimer**: This tool is for educational purposes only and should **not** be used as a substitute for professional medical advice or diagnosis.

---


---

## 🧠 What This App Does

- 📊 Predicts lung cancer **severity** using logistic regression
- 🎛️ Takes user input on lifestyle and clinical features (like age, smoking, diet)
- 📉 Visualizes EDA, hypothesis testing, and model evaluation
- 🧪 Deploys an actual trained ML model on the web

---

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Model**: Logistic Regression via scikit-learn
- **Deployment**: Streamlit Cloud

---

## 🧪 Machine Learning Pipeline

- `SimpleImputer()` for handling missing values  
- `StandardScaler()` for scaling features  
- `LogisticRegression(C=10)` with GridSearchCV tuning  
- Evaluation metrics:
  - ✅ Precision: ~94%
  - ✅ Recall: ~94%
  - ✅ F1-score: ~94%
  - ✅ Cross-validation used for stability

---

## To run this model in your local computer: 
pip install -r requirements.txt
streamlit run lungCancerSeverityApp.py


##💻 Author
Sujal Adhikari
📬 https://www.linkedin.com/in/sujaladhikari3/
🐙 GitHub: @suzaladhikari


