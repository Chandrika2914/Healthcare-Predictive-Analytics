# 🩺 Healthcare Predictive Analytics: Comparative Study of Supervised, Unsupervised & Dimensionality Reduction Techniques

### 📊 Project Overview
This project presents a **comprehensive machine learning analysis on healthcare data** to understand patient health indicators and predict outcomes such as **Hemoglobin**, **Glucose (GLU)**, and **Triglycerides**.  
It involves both **supervised** and **unsupervised** learning techniques — evaluating multiple regression algorithms, clustering approaches, and dimensionality reduction methods to identify the most effective modeling strategies.

⚠️ *Note:* The dataset used is confidential and cannot be shared publicly. This repository focuses on the analytical process, methodology, and comparative results.
---

## 🎯 Objectives
- Predict critical health indicators using supervised regression models.  
- Compare model performance across various algorithms using multiple evaluation metrics.  
- Explore unsupervised learning for clustering patients with similar conditions.  
- Apply dimensionality reduction techniques for better visualization and interpretation.  

---

## 🧩 Dataset Summary
- **Domain:** Healthcare  
- **Instances:** 1930  
- **Features:** 44  
- **Target Variables:** Hemoglobin, GLU (Glucose), Triglycerides  

### 🧼 Data Preprocessing
- Replaced missing values (NaN) with zeros or mean values.  
- Dropped irrelevant features (e.g., Date of Birth).  
- Applied **MinMaxScaler** for feature normalization.  
- Split dataset into **80% training** and **20% testing** sets.

---

## ⚙️ Workflow
1. Data collection and preprocessing  
2. Model development (Supervised learning)  
3. Hyperparameter optimization using **Grid Search**  
4. Evaluation using error metrics and accuracy  
5. Clustering analysis (Unsupervised learning)  
6. Dimensionality reduction and visualization  
7. Interpretation and conclusion  

---

## 🤖 Supervised Learning
### Algorithms Implemented
| Category | Algorithms |
|-----------|-------------|
| Baseline | Dummy Regressor |
| Tree-Based | Decision Tree, Random Forest |
| Ensemble | AdaBoost |
| Kernel-Based | Support Vector Machine (SVR) |
| Distance-Based | K-Nearest Neighbors (KNN) |
| Neural Network | Multilayer Perceptron (MLP) |

---

### 📈 Evaluation Metrics
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Percentage Error (MAPE)  
- Accuracy (100 - MAPE)

---

### 🧪 Key Results (Regression)
| Algorithm | MAE | MSE | RMSE | Accuracy (%) |
|------------|-----|-----|------|---------------|
| Decision Tree | 0.045 | 0.004 | 0.066 | 73.65 |
| Random Forest | 0.033 | 0.002 | 0.049 | **75.54** |
| Support Vector Regressor | 0.041 | 0.003 | 0.061 | 65.63 |
| KNN | 0.047 | 0.006 | 0.081 | 67.35 |
| AdaBoost | 0.037 | 0.002 | 0.052 | **75.81** |
| Neural Network | 0.056 | 0.006 | 0.082 | 56.54 |

✅ **Best Performing Models:**  
- **Random Forest (75.54%)**  
- **AdaBoost (75.81%)**  

### 📊 Predicted Health Features
| Feature | Random Forest | AdaBoost Grid Search |
|----------|----------------|----------------------|
| GLU | 75.54 | 75.81 |
| Hemoglobin | 86.21 | 84.41 |
| Triglycerides | 48.70 | 43.35 |

---

## 🔍 Unsupervised Learning
### Algorithms Used
- **K-Means Clustering**  
- **DBSCAN (Density-Based Spatial Clustering)**  

These algorithms were applied to identify natural clusters among patients based on health parameters.

### 📈 Cluster Evaluation
- Used **Silhouette Score** and **Elbow Method** to determine optimal number of clusters (K = 6).  
- **K-Means with PCA-reduced features** achieved best silhouette score of **0.718**, indicating well-separated clusters.  
- **K-tSNE** achieved a maximum silhouette score of **0.535**, showing moderate cluster quality.  
- **DBSCAN** after PCA reduction yielded a silhouette score of **0.195**, suggesting sparse clusters.  

---

## 🔻 Dimensionality Reduction
### Techniques Applied
| Method | Description | Purpose |
|--------|--------------|----------|
| **t-SNE** | Non-linear technique for visualizing high-dimensional data | Data exploration & visualization |
| **PCA (Principal Component Analysis)** | Linear transformation technique | Reducing dimensionality before clustering |

✅ PCA proved most effective for clustering, providing clearer cluster separation and improved silhouette score.

---

## 📊 Visual Results
- **Decision Tree & Random Forest Learning Curves** show reduced variance after Grid Search.  
- **t-SNE & PCA plots** clearly depict separability of patient groups.  
- **Silhouette & Elbow Curves** confirm optimal K for clustering (K=6).  

---

## 💡 Insights & Conclusion
- **Random Forest** and **AdaBoost** emerged as best supervised models, achieving the highest accuracy and lowest error.  
- **K-Means with PCA** produced the most coherent clusters for patient grouping.  
- **t-SNE visualizations** provided intuitive insights into high-dimensional data patterns.  
- Overall accuracy of **Random Forest** reached **86.21%** for Hemoglobin prediction — the highest among models tested.

---

## 🧰 Tools & Technologies
- **Programming:** Python  
- **Libraries:** Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn  
- **Techniques:** Supervised Learning, Unsupervised Learning, Dimensionality Reduction, Model Evaluation, Grid Search  

---

## 🧾 Skills Demonstrated
- Data cleaning and preprocessing  
- Regression and clustering modeling  
- Hyperparameter optimization (Grid Search)  
- Dimensionality reduction and visualization  
- Comparative performance analysis  
- Interpretation of ML metrics and results  
