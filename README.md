# 🎯 AI Customer Personality Analysis & Recommendation System

An interactive **Streamlit Web App** that uses **Unsupervised Machine Learning (K-Means Clustering)** to segment customers and generate intelligent marketing recommendations.

---

## 🚀 Project Overview

This application analyzes customer behavioral and demographic data to:

* 📊 Segment customers using **K-Means Clustering**
* 📈 Visualize clusters using **PCA**
* 🧠 Generate marketing recommendations based on cluster profiles
* 📉 Evaluate model quality using **Silhouette Score**

The system helps businesses understand customer personas and create targeted marketing strategies.

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Plotly**

---

## 📂 Features

✔ Upload custom cleaned CSV dataset
✔ Select number of clusters (2–8)
✔ Automatic scaling using StandardScaler
✔ K-Means clustering (k-means++)
✔ PCA 2D visualization
✔ Cluster distribution pie chart
✔ Dynamic marketing recommendations
✔ Silhouette score evaluation

---

## 📊 Expected Dataset Columns

Your dataset must include:

```
ID
Age
Income
Total_Spent
Enrollment_Days
Children
NumWebPurchases
NumCatalogPurchases
NumStorePurchases
NumWebVisitsMonth
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-link>
cd <your-project-folder>
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📌 How It Works

1. Upload your cleaned customer dataset.
2. Select number of clusters using sidebar slider.
3. App performs:

   * Data scaling
   * K-Means clustering
   * PCA transformation
4. View interactive cluster visualization.
5. Select a customer ID to generate marketing strategy.

---

## 🧠 Machine Learning Workflow

1. Feature Selection
2. Standard Scaling
3. K-Means Clustering
4. Silhouette Score Evaluation
5. PCA for Visualization

---

## 🎯 Use Cases

* Retail Customer Segmentation
* Marketing Campaign Personalization
* Business Intelligence Dashboards
* CRM Optimization
* Behavioral Analysis

---

## 🌐 Deployment

You can deploy this app easily on:

* Streamlit Cloud
* Render
* Railway
* Heroku

Make sure:

* `app.py`
* `requirements.txt`
* `README.md`

are in the root directory.

---

## 📈 Future Improvements

* Elbow Method Visualization
* Auto Persona Naming
* Download Clustered Dataset
* Model Persistence
* Advanced Recommendation Logic

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub and feel free to contribute!
