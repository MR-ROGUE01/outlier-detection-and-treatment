# 🎯 Outlier Detection and Treatment

A practical implementation of three widely used outlier handling techniques in Machine Learning.

---

## 📌 Problem

Outliers can distort statistical measures and negatively impact machine learning models.

In this project, the same dataset is analyzed using three different approaches:

* Z-Score Method
* IQR Method
* Percentile Capping (Winsorization)

---

## 📂 Project Structure

```text
outlier-detection-and-treatment/
│
├── 01_ZScore_Outlier_Detection.ipynb
├── 02_IQR_Outlier_Detection.ipynb
├── 03_Percentile_Capping_Winsorization.ipynb
│
├── weight-height.csv
└── README.md
```

---

# 1️⃣ Z-Score Method

### Formula

[
Z = \frac{X-\mu}{\sigma}
]

### Detection

```python
upper_limit = df['Height'].mean() + 3 * df['Height'].std()
lower_limit = df['Height'].mean() - 3 * df['Height'].std()
```

### Remove Outliers

```python
new_df = df[
    (df['Height'] < upper_limit) &
    (df['Height'] > lower_limit)
]
```

### Visualization

```python
sns.boxplot(df['Height'])
```

---

# 2️⃣ IQR Method

### Formula

[
IQR = Q3 - Q1
]

[
Lower = Q1 - 1.5(IQR)
]

[
Upper = Q3 + 1.5(IQR)
]

### Detection

```python
Q1 = df['Height'].quantile(0.25)
Q3 = df['Height'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
```

### Remove Outliers

```python
new_df = df[
    (df['Height'] >= lower_limit) &
    (df['Height'] <= upper_limit)
]
```

---

# 3️⃣ Percentile Capping (Winsorization)

### Calculate Limits

```python
upper_limit = df['Height'].quantile(0.99)
lower_limit = df['Height'].quantile(0.01)
```

### Remove Outliers

```python
new_df = df[
    (df['Height'] <= upper_limit) &
    (df['Height'] >= lower_limit)
]
```

### Capping

```python
df['Height'] = np.where(
    df['Height'] >= upper_limit,
    upper_limit,
    np.where(
        df['Height'] <= lower_limit,
        lower_limit,
        df['Height']
    )
)
```

---

## 📊 Visual Comparison

### Before Treatment

```python
sns.boxplot(df['Height'])
```

### After Treatment

```python
sns.boxplot(new_df['Height'])
```

---

## 🛠️ Libraries Used

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

---

## 🚀 What I Practiced

* Outlier Detection
* Outlier Removal
* Winsorization
* Statistical Analysis
* Data Visualization
* Data Preprocessing for ML

---

## 👨‍💻 Author

**Raj (MR-ROGUE01)**

B.Tech CSE (AI & ML)
