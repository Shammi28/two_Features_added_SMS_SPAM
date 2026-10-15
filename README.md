# SMS Spam Classification Using Support Vector Machines (SVM)

An end-to-end Natural Language Processing (NLP) and Machine Learning project that classifies SMS text messages as either **Ham** (legitimate) or **Spam**. 
The project leverages TF-IDF vectorization with n-grams and a Linear Support Vector Classifier (LinearSVC),
optimized using 5-fold cross-validation and hyperparameter tuning.

### 🌟 Additional Key Features

* **Visual Model Evaluation (Confusion Matrix & ROC-AUC Curve):**
  * **Confusion Matrix Display:** Provides a heatmap visualization showing the distribution of true vs. false predictions across Ham and Spam classes.
  * **ROC-AUC Analysis:** Evaluates classification performance across varying decision thresholds using the model's `decision_function`, computing both the ROC curve and the total AUC score.

* **Interactive Real-Time Testing Prompt:**
  * Uses dynamic text input (`input()`) to classify custom user SMS messages on demand.
  * Automatically pipes the entered text through the preprocessing (`clean_text`) and TF-IDF pipeline to display an immediate `[SPAM]` or `[HAM]` verdict.

📌 Project Overview

Unsolicited spam messages present privacy and security risks. This project demonstrates an automated pipeline to detect spam SMS messages using classical machine learning techniques:
* **Text Preprocessing:** Lowercasing, regex-based URL and email removal, punctuation and digit stripping, stopword filtering, and lemmatization.
* **Exploratory Data Analysis (EDA):** Class distribution inspection, message length analysis, and WordCloud visualization.
* **Feature Engineering:** TF-IDF (Term Frequency-Inverse Document Frequency) representation with unigrams and bigrams.
* **Model Training & Optimization:** Support Vector Classification (`LinearSVC`) tuned with `GridSearchCV`.
* **Evaluation & Visualizations:** Metrics reporting (Precision, Recall, F1-Score, Accuracy), Confusion Matrix display, and ROC-AUC curve analysis.
* **Interactive Inference:** Real-time user input prompt for testing custom SMS texts.


## 📊 Dataset

* **Source:** [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download)
* **Original Records:** 5,572 messages (4,825 Ham, 747 Spam)
* **Deduplication:** 403 duplicate messages removed, yielding 5,169 unique records
* **Target Encoding:**
  * `0` $\rightarrow$ Ham (Legitimate)
  * `1` $\rightarrow$ Spam

## 🛠️ Tech Stack & Libraries

* **Language:** Python
* **Data Processing & Analysis:** `pandas`, `numpy`
* **Natural Language Processing:** `nltk` (Stopwords, WordNetLemmatizer), `re`, `string`
* **Machine Learning:** `scikit-learn` (`Pipeline`, `TfidfVectorizer`, `LinearSVC`, `GridSearchCV`, `train_test_split`, `cross_val_score`)
* **Data Visualization:** `matplotlib`, `seaborn`, `wordcloud`

## ⚙️ Workflow & Architecture

```text
Raw SMS Text
     │
     ▼
Preprocessing (clean_text)
 ├── Lowercasing
 ├── Strip URLs & Email patterns
 ├── Remove numbers & punctuation
 ├── Remove NLTK English stopwords
 └── WordNet Lemmatization
     │
     ▼
Train / Test Split (80/20 Stratified)
     │
     ▼
Scikit-Learn Pipeline
 ├── TfidfVectorizer (max_features=5000, ngram_range=(1,2))
 └── LinearSVC (Tuned C=1.0)
     │
     ▼
Model Evaluation & Inference
 ├── Classification Report (Accuracy, Precision, Recall, F1)
 ├── Confusion Matrix & ROC-AUC Curve
 └── Real-time interactive testing interface

## 📈 Performance & Results

The model was evaluated on an unseen 20% stratified test set (1,034 messages):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | ~98.45% |
| **Precision (Spam)** | ~98.32% |
| **Recall (Spam)** | ~89.31% |
| **F1-Score (Spam)** | ~93.60% |
| **5-Fold Cross-Validation (Mean)** | ~97.95% |

### Detailed Classification Report
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       903
           1       0.98      0.89      0.94       131

    accuracy                           0.98      1034
   macro avg       0.98      0.95      0.96      1034
weighted avg       0.98      0.98      0.98      1034
