# 📧 Email Spam Prediction Using NLP

An end-to-end machine learning pipeline to classify emails as **Spam** or **Ham** using Natural Language Processing (NLP) techniques and Multinomial Naive Bayes.

---

## 🎯 Objective
Detect whether an email is **Spam** or **Ham** through text preprocessing, feature engineering (TF-IDF), and lightweight supervised learning.

---

## 📊 Dataset
- **Total Emails:** 5,171  
- **Class Distribution:** Ham = 3,672 | Spam = 1,499  
- **Label Encoding:** `ham → 0`, `spam → 1`  
- **Vector Space:** 37,803 unigram features (TF-IDF), CSR sparse matrix with 292,336 non-zero entries  

---

## 🧪 Approach
### Text Preprocessing
- Lowercasing, digit & punctuation removal, whitespace trimming  
- Removal of the token **“subject”**  
- **Porter Stemming** (NLTK) for normalization  

### Feature Engineering
- **TF-IDF Vectorization** with English stop-word removal  

### Train/Test Split & Class Imbalance Handling
- **80/20** split for train/test  
- **SMOTE** (`random_state=42`) applied only to training data  

### Model
- **Multinomial Naive Bayes** trained on TF-IDF vectors of oversampled data  

### Evaluation
- Tested on **untouched test set**  
- Metrics: Accuracy, Precision, Recall, F1-score  

---

## 📈 Results
**Test Set (n=1,035):**  
- **Accuracy:** 97%  
- **Ham (0):** Precision 0.99 | Recall 0.97 | F1 0.98  
- **Spam (1):** Precision 0.93 | Recall 0.98 | F1 0.96  

> High recall for spam ensures minimal missed detections.  
> High precision for ham reduces false positives.

---

## 🔍 Key Insights
- **Balanced Detection:** Prioritizes catching spam while avoiding false alarms.  
- **Effective Preprocessing:** Cleaning text and stemming significantly improved performance.  
- **Imbalance Solved:** SMOTE prevented bias toward majority class.  
- **Lightweight & Efficient:** TF-IDF + Naive Bayes gives excellent results with minimal complexity. 

---

## 🛠 Tech Stack
- Python  
- Pandas, NumPy  
- scikit-learn (TfidfVectorizer, MultinomialNB)  
- NLTK (PorterStemmer)  
- imbalanced-learn (SMOTE)  
- Matplotlib, Seaborn  

---
