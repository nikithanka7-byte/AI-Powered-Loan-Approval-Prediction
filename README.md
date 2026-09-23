#  AI-Powered Loan Approval Prediction

An AI-based web application that predicts whether a loan application is likely to be **Approved** or **Rejected** using Machine Learning. The application is built with **Python**, **Streamlit**,  providing a simple and interactive interface for users.

---

##  Project Overview

The Loan Approval Prediction System helps financial institutions and users estimate loan approval outcomes based on applicant information. The model is trained on historical loan data and uses machine learning to make predictions.

###  Features
-  User-friendly Streamlit web interface
-  Machine Learning-based loan approval prediction
-  Input parameters using sliders and dropdown menus
-  Instant prediction results
-  Pre-trained model (`model.pkl`) and scaler (`scaler.pkl`)

---

##  Technologies Used

- **Python**
- **Streamlit**
- **Pandas**
- **numpy**
- **Pickle**
- **Machine Learning**

---

##  Project Structure

```
AI-Powered-Loan-Approval-Prediction/
│
├── app.py                         # Streamlit web application
├── model.pkl                      # Trained machine learning model
├── scaler.pkl                     # Data scaler
├── loan_approval_dataset.csv      # Dataset used for training
├── Loan_Approval_Prediction.ipynb # Jupyter Notebook for model development
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

##  How to Run the Project

### 1️ Clone the Repository

```bash
git clone https://github.com/nikithanka7-byte/AI-Powered-Loan-Approval-Prediction.git
cd AI-Powered-Loan-Approval-Prediction
```

##  Input Features

The model predicts loan approval based on the following inputs:

| Feature | Description |
|----------|-------------|
| Number of Dependents | Total dependents of the applicant |
| Education | Graduated / Not Graduated |
| Self Employed | Yes / No |
| Annual Income | Applicant's annual income |
| Loan Amount | Requested loan amount |
| Loan Duration | Loan term in years |
| CIBIL Score | Applicant's credit score |
| Total Assets | Total value of applicant's assets |

---

##  Prediction Output

-  **Loan Approved**
-  **Loan Rejected**

The prediction is generated using the trained machine learning model after preprocessing the input data with the saved scaler.

---

## 📸 Application Preview


https://github.com/user-attachments/assets/d9b79d85-738b-42ca-b24c-a3b01d8558e9

<img width="1920" height="1080" alt="Screenshot (935)" src="https://github.com/user-attachments/assets/41a23cd4-e2f2-4679-a90c-78c2e433bfe4" />
<img width="1920" height="1080" alt="Screenshot (936)" src="https://github.com/user-attachments/assets/1e771807-9fab-47cf-82f6-a444195746ad" />
<img width="1920" height="1080" alt="Screenshot (937)" src="https://github.com/user-attachments/assets/205ce575-fcd5-4f30-98d2-9dffe7e39bdc" />
<img width="1920" height="1080" alt="Screenshot (938)" src="https://github.com/user-attachments/assets/6da4f9e2-2ff7-468c-aff8-42d1a8bc3554" />
<img width="1920" height="1080" alt="Screenshot (939)" src="https://github.com/user-attachments/assets/06de9dac-bb86-43da-9cfa-e6183baa5db0" />






---

##  Live Demo
** Live App:** https://4harpm83fe.preview.c40.airoapp.ai/?airoShareToken=2P9TpqmSgud0

---

##  Dataset

The project uses a loan approval dataset containing applicant details and loan status information. The data is preprocessed and used to train a classification model.



##  Author

**R. Nikitha**





