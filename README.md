

# HealthGuard AI – Diabetes & Heart Disease Risk Prediction Using Wearable Data

Real-time AI-powered health monitoring system that predicts **Diabetes Risk** and **Heart Disease Risk** using everyday wearable data:  
Blood Glucose · Heart Rate · SpO₂ · Steps · Sleep Duration & Quality · Workouts · Stress Levels

Live Demo (coming soon): [https://healthguard-ai.streamlit.app](https://healthguard-ai.streamlit.app)

---

### Project Goal

To build a **proactive, personalized health assistant** that:
- Detects early signs of **hypoglycemia/hyperglycemia**
- Predicts **cardiovascular risk** (arrhythmia, hypertension, heart strain)
- Analyzes **lifestyle patterns** from smartwatches/fitness bands
- Delivers **actionable, personalized wellness recommendations** in real time

> From raw wearable data to life-saving insights — no doctor visit required.

---

### Supported Health Parameters (Wearable-Compatible)

| Parameter             | Source                    | Used For                              |
|-----------------------|---------------------------|----------------------------------------|
| Blood Glucose         | CGM / Manual entry        | Diabetes risk & forecasting            |
| Heart Rate (BPM)      | Smartwatch / Band         | Cardiovascular risk, stress detection  |
| SpO₂ (Oxygen Saturation) | Smartwatch             | Hypoxia detection, sleep apnea risk    |
| Step Count            | Fitness band / Phone      | Activity level & metabolic health      |
| Sleep Duration & Stages | Smartwatch              | Recovery quality, insulin sensitivity  |
| Workout Intensity     | Smartwatch / App          | Exercise impact on glucose & heart     |
| Heart Rate Variability (HRV) | Advanced watches       | Stress & autonomic nervous system      |

---

### Core AI Models

| Task                          | Model Used           | Explainability     | Performance |
|-----------------------------|----------------------|--------------------|-----------|
| Diabetes Risk Classification | XGBoost + SMOTE      | SHAP Values        | 95.6% Accuracy |
| Glucose Forecasting (3–6h)   | LSTM (Deep Learning) | Time-series plot   | MSE ~1.4       |
| Heart Disease Risk           | XGBoost + Feature Eng| SHAP               | In Training    |
| Patient Behavioral Clustering| KMeans (5 clusters)  | Cluster insights   | Active         |
| Personalized Interventions   | Rule + Cluster-based | Natural language   | Live           |

---

### Tech Stack

| Category               | Tools & Libraries                                  |
|-----------------------|-----------------------------------------------------|
| Language              | Python 3.9+                                         |
| ML / DL               | XGBoost, TensorFlow/Keras, Scikit-learn             |
| Data Processing       | Pandas, NumPy                                       |
| Imbalanced Data       | SMOTE (imbalanced-learn)                            |
| Explainability        | SHAP (SHapley Values)                               |
| Clustering            | KMeans                                              |
| Dashboard             | Streamlit (Interactive Web App)                     |
| Visualization         | Plotly, Matplotlib, SHAP plots                      |
| Deployment            | Streamlit Community Cloud / Docker                  |

---

### Project Structure

```
.

├─ 📁.venv
├─ 📁core
│  ├─ 📁__pycache__
│  └─ 📄master.py
├─ 📁dashboard
│  ├─ 📁static
│  │  ├─ 📁sample-file
│  │  │  └─ 📄sample.csv
│  │  └─ 📄heart_ecg.png
│  ├─ 📁templates
│  │  ├─ 📄hiw.html
│  │  ├─ 📄index.html
│  │  ├─ 📄result.html
│  │  └─ 📄upload.html
│  ├─ 📁__pycache__
│  └─ 📄dashboard.py
├─ 📁models
│  ├─ 📁cardio
│  │  └─ 📄heart.py
│  ├─ 📁diabetes
│  │  ├─ 📁__pycache__
│  │  ├─ 📄diabetes.py
│  │  └─ 📄diabetes_model.pth
│  ├─ 📁__pycache__
│  └─ 📄__init__.py
├─ 📁uploads
├─ 📄.gitignore
├─ 📄app.py
├─ 📄README.md
└─ 📄requirements.txt
```

---

### How to Run Locally

```bash
# 1. Clone repo
git clone https://github.com/Divya-66/AI-ML-based-disease-prediction.git
cd AI-ML-based-disease-prediction

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the AI Health Dashboard
streamlit run app.py
```

Open http://localhost:8501 — upload your data and get instant insights!

---

### Sample Personalized Output

> **You belong to Cluster 4 – "Night Owls with High Glucose Variability"**  
> **Diabetes Risk**: Moderate (75th percentile)  
> **Heart Risk**: Elevated resting HR detected  
> **Recommendations**:  
> - Eat dinner 3+ hours before bed  
> - 20-min walk after meals  
> - Aim for 7.5+ hours sleep before 12 AM  
> - Monitor SpO₂ during sleep (possible apnea)

---

### Future Roadmap

| Feature                        | Status     |
|-------------------------------|------------|
| Apple Watch / Fitbit API integration | Planned    |
| Real-time push notifications  | In Progress |
| Doctor + Patient dual dashboard | Planned    |
| Heart Attack / Hypoglycemia alerts | Next       |
| Mobile App (iOS/Android)      | Planned    |
| Multi-language support         | Planned    |

---

### Dataset Sources

- Diabetes data: OHDSI / UCI Diabetes Dataset
- Heart & wearable data: Simulated + real anonymized logs
- All data used is fully anonymized and for research only

---

### Contributing

We welcome contributions!  
Whether it's improving models, adding new wearable features, or enhancing the UI — your help makes healthcare better.

---

### License

MIT License – Free to use, modify, and distribute.

---

**Built with care for a healthier tomorrow**  
**Divya-66 & Team** © 2025

---


``` 

Let me know when you push it — I’ll be the first to star it!


