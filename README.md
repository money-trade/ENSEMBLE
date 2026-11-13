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

```

---

## การติดตั้ง (Installation)

1. **โคลนโปรเจกต์ลงเครื่อง (หากใช้งานผ่าน Git):**
```bash
git clone <repository-url>
cd <repository-folder>

```


2. **ติดตั้งไลบรารีที่จำเป็น:**
```bash
pip install numpy pandas scikit-learn joblib

```



---

## วิธีการใช้งาน (Usage)

สามารถรันสคริปต์เพื่อทดสอบและเปรียบเทียบประสิทธิภาพของแต่ละอัลกอริทึมได้ผ่าน Terminal:

### 1. ทดสอบ Parallel Bagging Classifier

```bash
python 1_bagging_parallel.py

```

> **การทำงาน:** โหลดข้อมูล ➡️ แบ่ง Train/Test (70/30) ➡️ ทำ Standard Scaling ➡️ เทรนโมเดลผสมผสานแบบขนาน (`n_jobs=-1`) ➡️ แสดงผล **Accuracy, Confusion Matrix** และ **Classification Report**

### 2. ทดสอบ Sequential Boosting Classifier (SAMME)

```bash
python 2_boosting_sequential.py

```

> **การทำงาน:** โหลดและเตรียมข้อมูล ➡️ เทรนโมเดลทีละลำดับพร้อมคำนวณค่า *Weighted Error* และค่า *Alpha ($\alpha$)* ในแต่ละรอบ ➡️ สรุปผลการประเมินประสิทธิภาพโมเดล

---

## ภาพรวมการทำงานของอัลกอริทึม

| คุณสมบัติ | Parallel Bagging (`1_bagging_parallel.py`) | Sequential Boosting (`2_boosting_sequential.py`) |
| --- | --- | --- |
| **รูปแบบการเทรน** | ขนานพร้อมกัน (Parallel via `joblib`) | ต่อเนื่องทีละโมเดล (Sequential) |
| **การสุ่มข้อมูล** | Bootstrap Sampling (สุ่มแบบใส่คืน) | ใช้ข้อมูลทั้งหมดโดยปรับ Sample Weights |
| **อัลกอริทึมหลัก** | Majority Voting | Multi-class SAMME |
| **เป้าหมายหลัก** | ลด Variance (ป้องกัน Overfitting) | ลด Bias (เรียนรู้จากจุดที่ทายผิดซ้ำๆ) |

```

### สิ่งที่ปรับปรุงเพิ่มเติม:
1. **เพิ่ม Badges & Header:** ทำให้เปิดหน้า Repo มาแล้วดูทันสมัยและรู้ทันทีว่าใช้ Stack อะไรบ้าง
2. **จัดกลุ่ม Key Features:** แยกจุดเด่นของโค้ดออกมาให้อ่านง่ายขึ้น แทนการเอาไปรวมกับคำอธิบายไฟล์ทั้งหมด
3. **เปลี่ยนโครงสร้างเป็น Code Tree:** ใช้สัญลักษณ์ `├──` และ `└──` ซึ่งเป็นมาตรฐานสากลของ README
4. **เพิ่มตารางเปรียบเทียบ (Comparison Table):** ช่วยให้คนที่เข้ามาดูเข้าใจความแตกต่างระหว่างไฟล์ `1_bagging_parallel.py` และ `2_boosting_sequential.py` ได้ในพริบตา

<ElicitationsGroup message="ต้องการปรับแต่งส่วนไหนเพิ่มเติมไหมครับ?">
  <Elicitation label="แปล README ฉบับนี้เป็นภาษาอังกฤษทั้งหมด" query="ช่วยแปล README ฉบับที่ปรับปรุงใหม่นี้เป็นภาษาอังกฤษแบบเป็นมืออาชีพสำหรับอัปโหลดขึ้น GitHub"/>
  <Elicitation label="เพิ่มส่วนตัวอย่างผลลัพธ์ (Sample Output) ใน README" query="ช่วยเพิ่มหัวข้อตัวอย่างผลลัพธ์ (Sample Output) ของการรันทั้งสองไฟล์ลงใน README ให้ดูสมจริง"/>
  <Elicitation label="สร้างไฟล์ requirements.txt ให้เข้ากับโปรเจกต์" query="ช่วยเขียนเนื้อหาสำหรับไฟล์ requirements.txt ของโปรเจกต์นี้และปรับคำสั่งติดตั้งใน README ให้ใช้ pip install -r requirements.txt"/>
</ElicitationsGroup>

```
