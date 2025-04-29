
---

# 💼 OutText_Preprocessing Library – Outlier & Text Preprocessing Master Toolkit 🌟🧠🧹

Welcome to **OutText_Preprocessing** — your all-in-one Python library for advanced **outlier handling** and **text preprocessing**! 🎯 Whether you're building models for NLP 🧠 or machine learning 📈, this package is here to supercharge your data cleaning workflows 🚀.

---

## ✨ What’s Inside?

### 🧹 **1. Text Preprocessing Module**
Powerful functions to clean and transform raw text into model-ready input 💬⚙️.

### 🔍 **2. Outlier Removal & Impact Reduction**
A full suite of techniques to detect, reduce, or eliminate outliers for better model performance 📊🛡️.

---

## 🚀 Key Features

### 🧹 Text Preprocessing 📝

- ✅ Clean noisy text (emojis, links, HTML, hashtags)
- ✨ Normalize casing, fix spelling, remove symbols
- 🌐 Multilingual support
- 📊 Easily apply on Pandas columns

### 🔧 Outlier Handling 🧪

- 📉 Z-score removal and capping
- 🔬 Yeo-Johnson transformation
- 🌈 Impact reduction & smooth capping
- ✂️ Adaptive trimming (IQR method)
- 🔄 Local standardization with rolling window

---

## 📦 Installation

```bash
pip install OutText_preprocessing
```

---

## 🧠 Example Usage

```python
from OutText_preprocessing.text_cleaner import TextCleaner
from OutText_preprocessing.outlier_removal import OutlierRemover

# Text preprocessing
cleaner = TextCleaner()
clean_text = cleaner.clean("Th!s text 😜 has lots of NOISE!!! Check: https://example.com")

# Outlier removal
remover = OutlierRemover(method='impact_reduction')
processed_df = remover.fit_transform(your_dataframe)
```

---

## 🔁 Multi-Method Column-Wise Outlier Cleaning

```python
methods_columns_dict = {
    "zscore_capper": ["col1", "col2"],
    "impact_reduction": ["col3"],
}
processed_data = remover.multi_outlier_multi_columns(df, methods_columns_dict)
```

---

## 📘 Full Functionality

### ✅ `TextCleaner` Functions

- `clean()` – Clean general text
- `remove_emojis()`
- `remove_links_mentions_hashtags()`
- `extract_top_words()`
- Works directly on DataFrame columns 🐼

### 🧪 `OutlierRemover` Methods

- `fit_transform(data)`
- `multi_outlier_multi_columns(data, methods_columns_dict)`
- `get_available_methods()`

---

## 🧪 Performance Test 🏆

📊 When evaluated on synthetic classification tasks with injected outliers:

- **RobustScaler** → `0.9033` accuracy
- **OutText_Preprocessing (multi-method)** → `0.9067` accuracy 🔥

✅ Proven better or equal performance to standard scalers! 🎯

---

## 📚 Why Use This Library?

✨ One library for **both numeric and text preprocessing**

🧪 **Customizable** outlier thresholds, smoothing, rolling window

🧼 Super clean & fast **text cleaning** with regex, language support, emoji filtering

📈 Improve model accuracy in real-life dirty data scenarios

---

## 🔗 My Work & Socials

- 🏅 **Kaggle**: [anuragbanerjeeofficial](https://www.kaggle.com/anuragraj03/code)
- 💻 **GitHub**: [Anurag Raj](https://github.com/Anurag-raj03)
- 💼 **LinkedIn**: [Anurag Raj](https://www.linkedin.com/in/anurag-raj-770b6524a/)

- 

---


### ✅ **Method Usage Guide by Problem Type**

| Method Name               | Problem Type                                                                 | Suitability / Use Case                                                                                   |
|---------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| **`zscore`**              | Strict outlier **removal** in normally distributed data                      | Ideal for **cleaning data** by **removing extreme outliers**. Can cause **row loss**.                    |
| **`zscore_capper`**       | Moderating extreme values in normal data                                     | Use when you want to **retain data** but reduce influence of outliers via **capping**.                   |
| **`yeo_johnson`**         | Skewed data with non-Gaussian distribution                                   | Good for transforming **non-normal distributions** and **removing outliers** (but removes rows).         |
| **`yeo_johnson_capper`**  | Skewed data, need smooth transformation without row deletion                 | Same as above but **caps** values instead of removing. Great when **data retention** is critical.        |
| **`impact_reduction`**    | Large values dominate analysis but are valid                                 | Retain all data, but **minimize outlier impact**. Use in **regression and ML pre-processing**.           |
| **`adaptive_trimming`**   | Data with unknown distribution or extreme outliers                           | Based on **IQR**, robust to non-normal data. Useful in **robust statistics** or **median-based methods**.|
| **`smooth_capping`**      | Smoothly adjusting values instead of hard-capping or removing                | Best when a **softer influence reduction** is desired, good for **time-series or financial data**.       |
| **`local_standardization`** | Data with **seasonal trends, patterns, or local anomalies**                 | Useful for **time-series or rolling-window** data. Handles **local outliers** effectively.               |

---

### 🧠 Tips for Choosing the Right Method:

- For **statistical analysis** (like t-tests, regression): prefer **zscore_capper** or **impact_reduction**.
- For **ML preprocessing**: prefer **yeo_johnson_capper**, **adaptive_trimming**, or **smooth_capping**.
- For **time-series**: use **local_standardization** or **smooth_capping**.
- To **remove** bad data: use **zscore** or **yeo_johnson** (be aware of row loss).
---
