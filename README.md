# AI Skin Health Monitoring System

An AI-powered system that analyzes **skin images and lifestyle factors** to predict skin condition and provide health recommendations.

The system combines **Computer Vision (CNN)** and **Machine Learning (Random Forest)** with a **FastAPI backend and React frontend**.

---

# Features

- Skin image analysis using **Convolutional Neural Network (CNN)**
- Lifestyle-based risk prediction using **Random Forest**
- AI-based recommendations for improving skin health
- Interactive **React dashboard**
- REST API built with **FastAPI**

---

# System Architecture

User uploads skin image  
↓  
CNN Model (PyTorch)  
↓  
Skin Condition Prediction  
↓  
Lifestyle Inputs (sleep, stress, water, exercise, screen time)  
↓  
Random Forest Model  
↓  
Risk Level Prediction  
↓  
AI Recommendation  
↓  
Displayed on React Dashboard

---

# Tech Stack

### AI / Machine Learning
- PyTorch (CNN)
- Scikit-learn (Random Forest)

### Backend
- FastAPI
- Python

### Frontend
- React (Vite)
- CSS

### Data Processing
- Pandas
- NumPy

---

# Project Structure

```
skin-health-ai
│
├── backend
│   ├── app
│   │   └── api.py
│   ├── models
│   │   ├── train.py
│   │   └── lifestyle_model.pkl
│   └── data_processing
│
├── frontend
│   └── React dashboard
│
├── README.md
└── .gitignore
```

---

# Installation

## 1 Clone the repository

```
git clone https://github.com/Siddharth-AIML/skin-health-ai.git
cd skin-health-ai
```

---

## 2 Setup backend

```
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run backend server:

```
uvicorn app.api:app --reload
```

Backend runs on:

```
http://127.0.0.1:8000
```

---

## 3 Setup frontend

```
cd frontend
npm install
npm run dev
```

Frontend runs on:

```
http://localhost:5173
```

---

# Model Details

## CNN Model

Used for **skin image classification**

Classes:

- Healthy
- Mild
- Moderate

Framework used:

```
PyTorch
```

---

## Lifestyle Model

Used to predict **risk level** based on lifestyle inputs.

Features used:

- Sleep hours
- Stress level
- Water intake
- Exercise
- Screen time
- Skin condition (CNN output)

Algorithm:

```
Random Forest Classifier
```

---

# Example Output

```
Skin Condition: Mild

Confidence Score: 92%

Risk Level: High

Recommendation:
Increase sleep duration, reduce stress, and improve hydration.
```

---

# Future Improvements

- Add explainable AI (feature importance visualization)
- Improve CNN accuracy with larger dataset
- Deploy system using Docker
- Add mobile support

---

# Author

Siddharth Metkari  

AI / Machine Learning Enthusiast

GitHub:  
https://github.com/Siddharth-AIML

---

⭐ If you like this project, give it a star on GitHub.
