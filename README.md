# 🏦 Loan Eligibility Prediction – Streamlit App

A machine learning–powered Streamlit web application that predicts customer **loan approval status** based on financial and demographic attributes.  
The app includes **data upload, EDA, SMOTE balancing, Random Forest model training, evaluation, predictions, and summary reports**.

---

## 🚀 Features

✔ Upload CSV dataset  
✔ Automatic data cleaning & encoding  
✔ Exploratory Data Analysis (EDA)
- Loan status distribution  
- CIBIL score distribution  
- Boxplots (Income vs Status, CIBIL vs Status)  
- Correlation heatmap  

✔ ML Pipeline  
- SMOTE oversampling  
- Random Forest model  
- Train/test split  
- Performance metrics (Accuracy, Precision, Recall, F1)
- Confusion Matrix  
- Classification Report  

✔ Final loan approval predictions  
✔ Summary of approved vs rejected customers  


## 📌 Dataset Requirements

Your CSV must contain these columns:

- Customer_ID  
- Gender  
- Marital_Status  
- Education  
- Annual_Income  
- Bank_Balance  
- Cibil_Score  
- Gold_Assets  
- Land_Assets  
- Existing_Loan  
- Loan_Status  

---

## 🧠 Machine Learning Model

This app uses:

- **Random Forest Classifier**  
- Balanced training using **SMOTE**  
- One-Hot Encoding for categorical features  
- Ordinal Encoding for education level  

---

## 📊 Output

The app provides:

- Model accuracy summary  
- Confusion matrix heatmap  
- Classification report  
- Final loan decision table  
- Approval probability ranking  
- Summary report of approvals vs rejections  

---

## 🤝 Contributing

Pull requests and suggestions are welcome.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

Developed by **Parthiban**  
(Feel free to update your details here)

