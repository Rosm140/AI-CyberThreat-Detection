# 🛡️ AI Cyber Threat Detection System

## 📌 Project Overview

The **AI Cyber Threat Detection System** is a machine learning–based application designed to detect and classify cyber attacks from network traffic data.
The system is trained using the **CICIDS2017 dataset**, a well-known benchmark dataset for intrusion detection.

This project demonstrates how artificial intelligence can be applied in **cybersecurity** to automatically identify malicious activities such as DoS, DDoS, brute-force attacks, port scanning, and more.

---

## 🎯 Problem Statement

Traditional cybersecurity systems rely heavily on rule-based detection, which fails to detect new or unknown attacks.
With the increasing volume and complexity of network traffic, there is a need for **intelligent, data-driven security systems** that can detect attacks in real time.

This project aims to:

* Automatically detect cyber threats
* Classify different types of network attacks
* Reduce dependency on manual rule creation

---

## 🧠 Solution Approach

The proposed system uses **machine learning classification techniques** to analyze network flow features and predict whether the traffic is normal or malicious.

### Key steps:

1. Data collection from CICIDS2017 dataset
2. Data preprocessing and feature scaling
3. Training a machine learning model
4. Evaluating model performance
5. Deploying the trained model with an interactive UI

---

## 📊 Dataset Used

* **Dataset Name:** CICIDS2017
* **Source:** Canadian Institute for Cybersecurity
* **Type:** Network traffic flow data
* **Number of Features:** 52
* **Attack Categories:**

  * Normal Traffic
  * DoS
  * DDoS
  * Brute Force
  * Port Scanning
  * Web Attacks
  * Bots

---

## ⚙️ Machine Learning Model

* **Algorithm Used:** Random Forest Classifier
* **Why Random Forest?**

  * High accuracy
  * Handles large feature sets well
  * Robust to overfitting
* **Optimization:**

  * A lightweight version of the trained model is used for deployment to improve speed and reduce memory usage.

---

## 🖥️ System Architecture

```
User Input (Network Features)
        ↓
Data Scaling & Preprocessing
        ↓
Trained ML Model
        ↓
Attack Prediction + Confidence Score
        ↓
Streamlit User Interface
```

---

## 🧪 Testing & Evaluation

* The model was evaluated using:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
* Achieved **high accuracy (~99%)** on test data.
* Predictions change dynamically based on input feature values, demonstrating real-time inference.

---

## 🚀 Deployment & Execution

### Local Deployment (Recommended)

The project is deployed **locally using Streamlit**, ensuring stability and performance without cloud limitations.

### Steps to Run Locally:

```
venv\Scripts\activate
streamlit run app.py
```

The application runs on:

```
http://localhost:8501
```

---

## 🧩 Features of the Application

* Interactive web interface
* Real-time attack prediction
* Confidence score for predictions
* Demonstrates complete ML pipeline
* Suitable for academic demonstrations

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Libraries:** NumPy, Scikit-learn, Joblib, Streamlit
* **Dataset:** CICIDS2017
* **Tools:** GitHub, PowerShell

---

## 📁 Project Structure

```
AI_CyberThreat_Detection/
│
├── app.py                  # Streamlit application
├── requirements.txt        # Required libraries
├── model/
│   ├── cyber_model_small.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   └── label_encoder.pkl
├── notebooks/              # Training & preprocessing scripts
├── README.md
└── venv/
```

---

## 📌 Conclusion

This project successfully demonstrates the application of **machine learning in cybersecurity**.
By analyzing network traffic data, the system can accurately detect and classify cyber threats, helping improve network security and response time.

---

## 🔮 Future Enhancements

* CSV file upload for batch prediction
* Real-time network traffic monitoring
* Binary classification (Normal vs Attack)
* Model optimization using deep learning
* Cloud deployment with optimized resources

---

## 👨‍💻 Author

Rohit Mahadane
B.tech AIDS Student
AI & Cybersecurity Project
