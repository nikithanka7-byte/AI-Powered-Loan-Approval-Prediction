#  AI-Powered Loan Approval Prediction

A Streamlit web app that predicts whether a loan application will be **Approved** or **Rejected** using a Random Forest classifier.

---

##  Features

- Simple Streamlit interface with sliders and dropdowns
- Instant prediction with confidence score
- Pre-trained model (`model.pkl`) and scaler (`scaler.pkl`)
- Auto-retrains from the dataset if the saved model can't be loaded

---

##  Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- scikit-learn (`RandomForestClassifier`, `StandardScaler`)

---

##  Model

| Item | Details |
|------|---------|
| Algorithm | Random Forest Classifier (100 trees, `random_state=42`) |
| Preprocessing | `StandardScaler` |
| Dataset | 4,269 loan applications |
| Split | 80% train / 20% test |
| Test accuracy | **97.5%** |

---

##  Input Features

| Feature | Description |
|---------|-------------|
| Number of Dependents | Total dependents of the applicant |
| Education | Graduated / Not Graduated |
| Self Employed | Yes / No |
| Annual Income | Applicant's yearly income |
| Loan Amount | Requested loan amount |
| Loan Duration | Loan term in years |
| CIBIL Score | Credit score (300–900) |
| Total Assets | Combined residential, commercial, luxury and bank assets |

---

##  Project Structure

```
AI-Powered-Loan-Approval-Prediction/
├── app.py                          # Streamlit application
├── model.pkl                       # Trained Random Forest model
├── scaler.pkl                      # Fitted StandardScaler
├── loan_approval_dataset.csv       # Training dataset
├── Loan_Approval_Prediction.ipynb  # Model development notebook
├── requirements.txt                # Dependencies
└── README.md
```

---



---

##  Application Preview


https://github.com/user-attachments/assets/4d790a70-e7f5-448d-8ab5-3032967c13ea



<img width="1362" height="639" alt="image" src="https://github.com/user-attachments/assets/75b3da85-3f31-4dc5-adfb-8fe66caec3fb" />
<img width="1366" height="633" alt="image" src="https://github.com/user-attachments/assets/5f281508-aa77-4b06-86d4-334265d59f89" />
<img width="1365" height="624" alt="image" src="https://github.com/user-attachments/assets/5bc934a5-c591-4662-b8c6-2019a5007da6" />
<img width="1351" height="627" alt="image" src="https://github.com/user-attachments/assets/b84886d7-465e-4bb7-9f6e-095d65f47f7a" />
<img width="1332" height="561" alt="image" src="https://github.com/user-attachments/assets/d8e018b9-c379-4358-a815-b747eff6048f" />










---

##  Live Demo

**Live App:** https://ai-powered-loan-approval-prediction-pgeebgxdnxvdyqzqcaqhuk.streamlit.app/

---

##  Author

**R. Nikitha**
