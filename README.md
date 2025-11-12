# SMS Spam Classification using NLP: Achieving 100% Precision

## Overview
This project built a highly reliable SMS spam classifier using Natural Language Processing (NLP) techniques and the Multinomial Naive Bayes algorithm. The main goal was to achieve the highest possible **precision** for the 'spam' class.

## Steps Followed

### 1. Data Cleaning and Preparation
* Loaded and inspected the raw dataset.
* **Data Cleaning:** Removed redundant, mostly null columns and renamed the main columns to `labels` and `message`.
* **Target Encoding:** Converted categorical labels (`ham`, `spam`) into numerical form ($0$, $1$).

### 2. Advanced Text Preprocessing
A custom pipeline was created to clean text while protecting important information:
* **Noise Handling:** Used a regular expression to clean text, but made sure to **keep numbers and special characters** (`!`, `$`, `%`, etc.) because they are common signs of spam.
* **Smart Lower-casing (SpaCy-based):** Used **spaCy's entity and POS tagging** to only change non-entity words to lower-case. This maintains the meaning of Proper Nouns and Acronyms.
* **Normalization:** Applied **WordNetLemmatizer** to reduce words to their base form and removed common **Stopwords**.

### 3. Feature Engineering and Optimization
* **Feature Extraction:** Used **TF-IDF (Term Frequency–Inverse Document Frequency)** to turn the text into a matrix of weighted features.
* **Hyperparameter Tuning:** Tuned the `max_features` parameter in `TfidfVectorizer` to find the best vocabulary size for the model.

### 4. Model Training and Evaluation
* **Model Choice:** Selected **Multinomial Naive Bayes** because it is effective for text classification and works well with class imbalance.
* **Optimal Configuration:** The model trained with "max_features = 3300" provided the best result: **100% Precision** for the spam class.

## Learnings

* **Precision Over Recall for Spam:** In spam filtering, getting a False Positive (blocking a good message) is a bigger problem than a False Negative (missing a spam message). The final model prioritized **Precision = 100%** to solve this.

* **The Value of Context in Cleaning:** The custom, spaCy-driven lower-casing approach improved performance by protecting words that carry important contextual meaning.

* **Deployment Assets:** Saved the final **`MultinomialNB` model** and the trained **`TfidfVectorizer`** using `joblib`, which is essential for consistent deployment.

## Tech Stack
* **Python**
* **NLTK, spaCy** (for advanced NLP preprocessing)
* **Scikit-learn** (Multinomial Naive Bayes, TF-IDF, Metrics)
* **NumPy, Pandas**
* **joblib** (for model saving)


https://github.com/user-attachments/assets/ba47f1e5-e47e-432b-a0a9-9d1b1f5755ee

