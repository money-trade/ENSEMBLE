import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


def load_data(path="Iris.csv"):
    """โหลดจาก CSV ถ้ามีไฟล์ ไม่งั้นใช้ Iris ของ sklearn"""
    if os.path.exists(path):
        df = pd.read_csv(path).drop(columns=["Id", "id", "ID"], errors="ignore")
        y, names = pd.factorize(df.iloc[:, -1], sort=True)
        return df.iloc[:, :-1].values, y, list(names)
    iris = load_iris()
    return iris.data, iris.target, list(iris.target_names)


class ParallelBagging:
    def __init__(self, estimators, n_estimators=15, n_jobs=-1, random_state=42):
        self.estimators, self.n_estimators = estimators, n_estimators
        self.n_jobs, self.random_state = n_jobs, random_state

    @staticmethod
    def _fit_one(est, X, y, seed):
        # Bootstrap: สุ่มแบบใส่คืน ขนาดเท่าข้อมูลเดิม
        idx = np.random.RandomState(seed).choice(len(X), len(X), replace=True)
        return clone(est).fit(X[idx], y[idx])

    def fit(self, X, y):
        seeds = np.random.RandomState(self.random_state).randint(0, 100000, self.n_estimators)
        self.models_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one)(self.estimators[i % len(self.estimators)], X, y, s)
            for i, s in enumerate(seeds)
        )
        return self

    def predict(self, X):
        # Majority voting
        preds = np.array([m.predict(X) for m in self.models_])
        return np.apply_along_axis(lambda p: np.bincount(p).argmax(), 0, preds)


if __name__ == "__main__":
    X, y, names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    model = ParallelBagging([
        DecisionTreeClassifier(max_depth=4, random_state=42),
        KNeighborsClassifier(),
        SVC(random_state=42),
        LogisticRegression(max_iter=200),
        GaussianNB(),
    ]).fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=names))