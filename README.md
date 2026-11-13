# Ensemble Learning: Custom Bagging & Boosting (Iris Dataset)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)

โปรเจกต์พัฒนาอัลกอริทึม **Ensemble Learning** ทั้งรูปแบบ **Bagging** และ **Boosting** ขึ้นมาใหม่จากศูนย์ (Custom Implementation from Scratch) บนชุดข้อมูลดอกไม้ **Iris** โดยผสานการทำงานของ Base Classifiers ที่หลากหลาย ได้แก่ *Decision Tree, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Logistic Regression* และ *Naive Bayes* 

---

## จุดเด่นของโปรเจกต์ (Key Features)

* **Heterogeneous Base Learners:** ผสมผสานโมเดลพื้นฐานหลายประเภทเข้าด้วยกันในชุด Ensemble เดียว
* **Custom Parallel Bagging:** จำลองการสุ่มตัวอย่างแบบ *Bootstrap Sampling* พร้อมรันแบบขนาน (Parallel Computing) ผ่าน `joblib` และตัดสินผลด้วย *Majority Voting*
* **Custom Sequential Boosting:** พัฒนาอัลกอริทึม **Multi-class SAMME** ที่เรียนรู้จากความผิดพลาดในรอบก่อนหน้าด้วยการปรับค่าน้ำหนักข้อมูล (Sample Weighting)
* **Auto-Fallback Dataset:** รองรับการโหลดข้อมูลจากไฟล์ `Iris.csv` ในเครื่อง และดึงข้อมูลจาก `sklearn.datasets` อัตโนมัติหากไม่พบไฟล์

---

## โครงสร้างโปรเจกต์ (Project Structure)

```text
├── 1_bagging_parallel.py      # อัลกอริทึม Parallel Bagging (Bootstrap + Majority Voting)
├── 2_boosting_sequential.py   # อัลกอริทึม Sequential Boosting (Multi-class SAMME)
├── Iris.csv                   # ชุดข้อมูลดอกไม้ Iris (มีระบบสำรองอัตโนมัติหากไม่มีไฟล์)
├── .gitignore                 # ไฟล์ละเว้นแคชและไฟล์ชั่วคราวสำหรับ Git
└── README.md                  # เอกสารแนะนำการใช้งานโปรเจกต์
