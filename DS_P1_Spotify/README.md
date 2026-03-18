# 🎵 Spotify Popularity Prediction

An end-to-end Data Science project that predicts the likelihood of a song becoming popular based on its audio features, using Machine Learning and model interpretability techniques.

---

## 🚀 Live Demo

👉 [Streamlit App](https://56vwib9tbmappcg336np8b5.streamlit.app/)

---

## 📌 Problem Statement

The music industry invests heavily in production and promotion, yet predicting a song's success remains highly uncertain.

This project aims to answer:

> Can we predict a song’s popularity using only its audio features?

---

## 📊 Dataset

- Source: Spotify Audio Features Dataset  
- Observations: ~130,000 tracks  
- Features include:
  - Acousticness  
  - Danceability  
  - Energy  
  - Loudness  
  - Tempo  
  - Valence  
  - Key, Mode, Time Signature  

**Target Variable:**
- Binary classification: **Popular vs Not Popular**

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights:

- Tracks with higher **energy** and **loudness** tend to be more popular  
- **Acousticness** is generally higher in less popular tracks  
- **Danceability** and **tempo** show influence on engagement  

---

## 🛠️ Feature Engineering

- One-hot encoding applied to:
  - Key  
  - Mode  
  - Time Signature  

- Feature alignment to match model training structure  
- Consistent preprocessing between training and inference  

---

## 🤖 Modeling

**Model Used:**
- Random Forest Classifier  

**Why Random Forest?**
- Captures non-linear relationships  
- Robust to noise and outliers  
- Strong baseline for tabular data  

**Performance:**

- ROC-AUC: **0.72**

---

## 🔍 Model Explainability

To ensure transparency, SHAP (SHapley Additive Explanations) was implemented.

- Explains individual predictions  
- Identifies feature impact on model decisions  
- Waterfall plots show how each feature contributes to the final prediction  

---

## 🖥️ Deployment

The model is deployed using **Streamlit**:

- Interactive dashboard  
- Real-time predictions  
- User-controlled audio features  
- Integrated SHAP visualizations for interpretability  

---

## 📂 Project Structure

Spotify-Popularity-Predictor/
│
├── data/
├── notebooks/
├── models/
│ └── random_forest.pkl
├── app.py
├── requirements.txt
└── README.md---

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Miguel-sierra89/DS.git
   cd Spotify-Popularity-Predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## 🛠️ Technologies Used

- **Python** (Pandas, NumPy, Scikit-Learn)
- **Machine Learning** (Random Forest)
- **Model Interpretability** (SHAP)
- **Deployment** (Streamlit)
- **Visualization** (Matplotlib, Seaborn)
