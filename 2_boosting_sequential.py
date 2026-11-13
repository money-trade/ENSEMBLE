import os
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# นำเข้า Classifier หลายชนิดมาผสมกัน
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def load_iris_dataset(csv_path="Iris.csv"):
    """โหลดข้อมูล Iris จากไฟล์ CSV (หากไม่พบไฟล์จะดึงจาก sklearn อัตโนมัติ)"""
    if os.path.exists(csv_path):
        print(f"กำลังโหลดข้อมูลจากไฟล์: {csv_path}")
        df = pd.read_csv(csv_path)
        
        for col in ["Id", "id", "ID"]:
            if col in df.columns:
                df = df.drop(columns=[col])
                
        X = df.iloc[:, :-1].values
        y_raw = df.iloc[:, -1].values
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        target_names = [str(cls) for cls in le.classes_]
    else:
        print(f"ไม่พบไฟล์ '{csv_path}' ในโฟลเดอร์ ระบบใช้ชุดข้อมูล Iris มาตรฐานแทน")
        iris = load_iris()
        X, y = iris.data, iris.target
        target_names = [str(cls) for cls in iris.target_names]
        
    return X, y, target_names


class CustomSequentialBoosting:
    """อัลกอริทึม Multi-class Boosting (SAMME) ที่เขียนขึ้นเองแบบ Sequential"""
    
    def __init__(self, base_classifiers, n_estimators=15, learning_rate=1.0, random_state=42):
        self.base_classifiers = base_classifiers
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.models_ = []
        self.model_weights_ = []
        self.classes_ = None
        self.n_classes_ = 0

    def fit(self, X, y):
        """ฝึกสอนโมเดลต่อเนื่องกันแบบ Sequential โดยปรับถ่วงน้ำหนักข้อมูลในแต่ละรอบ"""
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # 1. กำหนดน้ำหนักเริ่มต้นของข้อมูลทุกตัวให้เท่ากัน (1 / N)
        sample_weights = np.full(n_samples, 1.0 / n_samples)
        
        self.models_ = []
        self.model_weights_ = []

        # 2. วนลูปฝึกสอนทีละโมเดลแบบ Sequential
        for m in range(self.n_estimators):
            base_est = self.base_classifiers[m % len(self.base_classifiers)]
            model = clone(base_est)
            
            # ฝึกโมเดลโดยส่ง sample_weight (หากโมเดลไม่รองรับ sample_weight จะใช้การสุ่มตามน้ำหนักแทน)
            try:
                model.fit(X, y, sample_weight=sample_weights)
            except TypeError:
                indices = rng.choice(n_samples, size=n_samples, replace=True, p=sample_weights)
                model.fit(X[indices], y[indices])

            # ทำนายผลบนชุดฝึกสอนเพื่อหาจุดที่ทายผิด
            y_pred = model.predict(X)
            incorrect = (y_pred != y).astype(float)

            # 3. คำนวณค่าความผิดพลาดถ่วงน้ำหนัก (Weighted Error)
            estimator_error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)
            estimator_error = np.clip(estimator_error, 1e-10, 1.0 - 1e-10)

            # 4. คำนวณน้ำหนักความสำคัญของโมเดล (Alpha) ตามสูตร Multi-class SAMME
            alpha = self.learning_rate * (
                np.log((1.0 - estimator_error) / estimator_error) + np.log(self.n_classes_ - 1.0)
            )

            # หากโมเดลมีความแม่นยำแย่กว่าการเดาสุ่ม ให้รีเซ็ตน้ำหนักข้อมูลใหม่
            if alpha <= 0:
                sample_weights = np.full(n_samples, 1.0 / n_samples)
                continue

            # 5. ปรับเพิ่มน้ำหนักให้กับข้อมูลที่ทายผิด เพื่อให้โมเดลถัดไปโฟกัสจุดที่ผิดมากขึ้น
            sample_weights *= np.exp(alpha * incorrect)
            
            # ปรับค่าน้ำหนักรวมให้มีผลรวมเท่ากับ 1 (Normalization)
            sample_weights /= np.sum(sample_weights)

            # บันทึกโมเดลและค่าน้ำหนักของโมเดลในรอบนี้
            self.models_.append(model)
            self.model_weights_.append(alpha)
            
            print(
                f"รอบที่ {m + 1:02d} | โมเดล: {model.__class__.__name__:<24} | "
                f"Weighted Error: {estimator_error:.4f} | Alpha: {alpha:.4f}"
            )

        return self

    def predict(self, X):
        """ทำนายผลด้วยการโหวตแบบถ่วงน้ำหนักตามค่า Alpha ของแต่ละโมเดล"""
        n_samples = X.shape[0]
        class_scores = np.zeros((n_samples, self.n_classes_))

        for model, alpha in zip(self.models_, self.model_weights_):
            preds = model.predict(X)
            for idx, cls in enumerate(self.classes_):
                class_scores[preds == cls, idx] += alpha

        return self.classes_[np.argmax(class_scores, axis=1)]


if __name__ == "__main__":
    # 1. โหลดข้อมูล (เปลี่ยนชื่อไฟล์ตรงนี้ได้หากไฟล์แนบชื่ออื่น เช่น 'iris.csv')
    X, y, target_names = load_iris_dataset("Iris.csv")

    # 2. แบ่งข้อมูล Train 70% / Test 30% และปรับสเกลข้อมูล
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. กำหนด Weak / Base Classifiers หลายชนิดมาผสมกัน
    mixed_classifiers = [
        DecisionTreeClassifier(max_depth=2, random_state=42),
        GaussianNB(),
        LogisticRegression(C=0.5, max_iter=200, random_state=42),
        SVC(kernel="linear", C=0.5, random_state=42),
        KNeighborsClassifier(n_neighbors=5)
    ]

    # 4. สร้างและฝึกสอนโมเดล Custom Sequential Boosting
    print("\n================ เริ่มการฝึกสอน Sequential Boosting ================")
    boosting_model = CustomSequentialBoosting(
        base_classifiers=mixed_classifiers,
        n_estimators=15,
        learning_rate=0.5,
        random_state=42
    )
    boosting_model.fit(X_train, y_train)

    # 5. ทดสอบและแสดงผลลัพธ์
    y_pred = boosting_model.predict(X_test)
    
    print("\n================ ผลลัพธ์ Sequential Boosting ================")
    print(f"จำนวนโมเดลที่ผ่านเกณฑ์: {len(boosting_model.models_)} โมเดล")
    print(f"ความแม่นยำ (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
