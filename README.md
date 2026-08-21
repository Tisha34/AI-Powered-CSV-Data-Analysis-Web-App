
# AI-Powered CSV Data Analysis Web App

Jupyter EDA is slow. This app skips the notebook entirely — upload a CSV and
get a data quality score, an EDA report, plain-English Q&A, and charts in the
browser. Backed by **LLaMA 3.3 70B via Groq API**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat&logo=flask)
![LLaMA](https://img.shields.io/badge/LLaMA-3.3_70B-orange?style=flat)
![Groq](https://img.shields.io/badge/Groq-API-red?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> 🔗 **Live Demo:** [Add deployed URL here]

---

## Demo

<!-- Replace with your actual GitHub-hosted demo.mp4 asset link -->
https://github.com/Tisha34/AI-Powered-CSV-Data-Analysis-Web-App/assets/222008937/demo.mp4

---

## Screenshots

**Upload & Dataset Summary**
![Upload](screenshots/upload.png)

**Data Quality Scorecard**
![Quality](screenshots/quality.png)

**Auto EDA Report**
![EDA](screenshots/eda1.png)
![EDA Heatmap](screenshots/eda2.png)

**Chart Generator**
![Chart](screenshots/chart.png)

---

## What it does

**Data Quality Check**
Scores the dataset from 0 to 100. Catches missing values, duplicate rows,
per-column outliers, and type mismatches — with notes on what to fix.

**Auto EDA**
One click. Produces a written summary of key findings, a correlation heatmap,
skewness flags, and a numeric column breakdown. Takes about 15 seconds.

**Ask Your Data**
Type a question in plain English — *"what is the average balance of customers
who churned?"* — and get an answer pulled from your actual data. Follow-up
questions stay in context, so you don't need to re-upload between questions.

**Chart Generator**
Describe the chart. LLaMA 3.3 70B writes the matplotlib code, the app renders
it, and you download a PNG.

---

## Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| AI Model | LLaMA 3.3 70B via Groq API |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib |
| Frontend | HTML, CSS, JavaScript |

---

## Run it locally

```bash
git clone https://github.com/Tisha34/AI-Powered-CSV-Data-Analysis-Web-App.git
cd AI-Powered-CSV-Data-Analysis-Web-App
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
```

Free key at https://console.groq.com

```bash
python app.py
```

Open `http://127.0.0.1:5000` and upload any CSV.

---

## Tested with

Churn Modelling dataset — 10,000 rows, 14 columns. Should work with any
well-structured CSV.

---

Tisha Gandhi — [LinkedIn](https://linkedin.com/in/tisha-gandhi) · [GitHub](https://github.com/Tisha34)