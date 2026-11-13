import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


class SequentialBoosting:
    """Multi-class Boosting (SAMME) เขียนเอง"""

    def __init__(self, estimators, n_estimators=15, learning_rate=1.0, random_state=42):
        self.estimators, self.n_estimators = estimators, n_estimators
        self.lr, self.random_state = learning_rate, random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        w = np.full(n, 1 / n)  # น้ำหนักเริ่มต้นเท่ากันทุกตัว
        self.models_, self.alphas_ = [], []

        for m in range(self.n_estimators):
            model = clone(self.estimators[m % len(self.estimators)])
            try:
                model.fit(X, y, sample_weight=w)
            except TypeError:  # โมเดลที่ไม่รับ sample_weight (เช่น KNN) → สุ่มข้อมูลตามน้ำหนักแทน
                idx = rng.choice(n, n, p=w)
                model.fit(X[idx], y[idx])

            wrong = model.predict(X) != y
            err = np.clip(w[wrong].sum(), 1e-10, 1 - 1e-10)  # weighted error
            alpha = self.lr * (np.log((1 - err) / err) + np.log(K - 1))

            if alpha <= 0:  # แย่กว่าเดาสุ่ม → รีเซ็ตน้ำหนักแล้วข้าม
                w = np.full(n, 1 / n)
                continue

            w *= np.exp(alpha * wrong)  # เพิ่มน้ำหนักตัวที่ทายผิด
            w /= w.sum()
            self.models_.append(model)
            self.alphas_.append(alpha)
            print(f"รอบ {m + 1:02d} | {type(model).__name__:<24} | error {err:.4f} | alpha {alpha:.4f}")
        return self

    def predict(self, X):
        # โหวตถ่วงน้ำหนักด้วย alpha
        scores = np.zeros((len(X), len(self.classes_)))
        for model, alpha in zip(self.models_, self.alphas_):
            scores += alpha * (model.predict(X)[:, None] == self.classes_)
        return self.classes_[scores.argmax(axis=1)]


if __name__ == "__main__":
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    model = SequentialBoosting([
        DecisionTreeClassifier(max_depth=2, random_state=42),
        GaussianNB(),
        LogisticRegression(C=0.5, max_iter=200),
        SVC(kernel="linear", C=0.5),
        KNeighborsClassifier(),
    ], learning_rate=0.5).fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nจำนวนโมเดลที่ใช้: {len(model.models_)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=iris.target_names))