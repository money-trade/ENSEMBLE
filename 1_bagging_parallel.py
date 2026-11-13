import os
from collections import Counter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# นำเข้า Classifier หลายชนิดมาผสมกัน
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


def load_iris_dataset(csv_path="Iris.csv"):
    """โหลดข้อมูล Iris จากไฟล์ CSV (หากไม่พบไฟล์จะดึงจาก sklearn อัตโนมัติ)"""
    if os.path.exists(csv_path):
        print(f"กำลังโหลดข้อมูลจากไฟล์: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # ลบคอลัมน์ Id ออกหากมีอยู่ในไฟล์ CSV
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


class CustomParallelBagging:
    """อัลกอริทึม Bagging ที่เขียนขึ้นเองและทำงานแบบ Parallel"""
    
    def __init__(self, base_classifiers, n_estimators=15, n_jobs=-1, random_state=42):
        self.base_classifiers = base_classifiers
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.models_ = []

    def _fit_single_estimator(self, estimator, X, y, seed):
        """ฟังก์ชันย่อยสำหรับสุ่ม Bootstrap และฝึกโมเดล 1 ตัว (รันแยกแต่ละ Process/Thread)"""
        rng = np.random.RandomState(seed)
        n_samples = X.shape[0]
        
        # 1. Bootstrap Sampling: สุ่มตัวอย่างขนาดเท่าเดิมแบบใส่คืน (replace=True)
        bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[bootstrap_indices]
        y_boot = y[bootstrap_indices]
        
        # 2. คัดลอกและฝึกสอนโมเดล
        model = clone(estimator)
        model.fit(X_boot, y_boot)
        return model

    def fit(self, X, y):
        """ฝึกสอนโมเดลทั้งหมดพร้อมกันแบบ Parallel"""
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 100000, size=self.n_estimators)
        
        # เลือก Classifier หมุนเวียนกันไปตามจำนวน n_estimators
        chosen_estimators = [
            self.base_classifiers[i % len(self.base_classifiers)]
            for i in range(self.n_estimators)
        ]
        
        # รันการฝึกสอนแบบ Parallel ด้วย joblib
        self.models_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_estimator)(est, X, y, seed)
            for est, seed in zip(chosen_estimators, seeds)
        )
        return self

    def predict(self, X):
        """ทำนายผลแบบ Parallel และโหวตเสียงข้างมาก (Majority Voting)"""
        # รันการทำนายของแต่ละโมเดลแบบ Parallel
        all_predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(model.predict)(X) for model in self.models_
        )
        
        # แปลงขนาดเป็น (จำนวนตัวอย่าง, จำนวนโมเดล)
        all_predictions = np.array(all_predictions).T
        
        # Majority Voting สำหรับแต่ละตัวอย่าง
        final_predictions = []
        for sample_preds in all_predictions:
            most_common_class = Counter(sample_preds).most_common(1)[0][0]
            final_predictions.append(most_common_class)
            
        return np.array(final_predictions)


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

    # 3. กำหนด Classifier หลากหลายชนิดมาผสมกัน
    mixed_classifiers = [
        DecisionTreeClassifier(max_depth=4, random_state=42),
        KNeighborsClassifier(n_neighbors=5),
        SVC(kernel="rbf", C=1.0, random_state=42),
        LogisticRegression(max_iter=200, random_state=42),
        GaussianNB()
    ]

    # 4. สร้างและฝึกสอนโมเดล Custom Parallel Bagging
    bagging_model = CustomParallelBagging(
        base_classifiers=mixed_classifiers,
        n_estimators=15,  # ใช้ 15 โมเดล (ชนิดละ 3 โมเดล)
        n_jobs=-1,        # ใช้ทุกแกน CPU รันแบบ Parallel
        random_state=42
    )
    bagging_model.fit(X_train, y_train)

    # 5. ทดสอบและแสดงผลลัพธ์
    y_pred = bagging_model.predict(X_test)
    
    print("\n================ ผลลัพธ์ Parallel Bagging ================")
    print(f"จำนวนโมเดลทั้งหมดที่ใช้โหวต: {len(bagging_model.models_)} โมเดล")
    print(f"ความแม่นยำ (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
