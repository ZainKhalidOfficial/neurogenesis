# 🧠 Neurogenesis

**Python Version:** 3.11.9  
**Status:** ⚠️ Pre-Alpha (Under Active Development)

Neurogenesis is a lightweight, educational machine learning library built from scratch in Python. Its primary aim is to focus on simplicity to aid in learning and to provide a platform for experimentation of unique ideas without having to face complex production-grade codebases of popular libraries which are highly optimized for performance.

> 🚧 **Disclaimer:**  
> Neurogenesis is currently in its early development stages. Expect incomplete features, bugs, and minimal optimization. Restructuring, refactoring, and formal documentation are planned for future releases.

---

## ✅ Algorithms Implemented (v0.0.1)

- K-Nearest Neighbors (KNN)
- Linear Regression
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM) – Binary Classification
- Decision Tree
- Random Forest
- Artificial Neural Networks (ANN) 

---

## 📌 Planned for v1.0.0

- KNN  
- Linear Regression  
- Logistic Regression  
- Naive Bayes  
- SVM  
- Decision Tree  
- Random Forest  
- K-Means Clustering  
- AdaBoost  
- Artificial Neural Networks (ANN) 
- Convolutional Neural Networks (CNN)  

---

## 🚀 Installation

Install Neurogenesis via pip:

```bash
pip install neurogenesis
```

## ⚙️ How to use
Here's a basic example using the RandomForest classifier:

```python 
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from neurogenesis.models import RandomForest

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForest(n_trees=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy:", np.sum(y_true == y_pred) / len(y_true))

```

🧪 Make sure your data (X_train, y_train, X_test, y_test) is preprocessed and in the correct format (e.g., NumPy arrays).

## 📄 License

This project is licensed under the **GNU General Public License v3.0**.  
You may redistribute, modify, and/or use the code under the terms of this license.

Please see the full license text online:  
🔗 https://www.gnu.org/licenses/gpl-3.0.en.html

> ⚠️ Note: Any derivative works must also be licensed under GPLv3 or a compatible license.

## 🙋‍♂️ Contributing

Contributions, bug reports, and feature requests are welcome!  
If you'd like to improve the library, fix a bug, or suggest a new algorithm, feel free to open an issue or submit a pull request.

📧 mzainkhalidofficial@gmail.com  
🔗 [GitHub Issues](https://github.com/ZainKhalidOfficial/neurogenesis/issues)

Thanks to the open-source community for continuous inspiration and learning!