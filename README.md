# 📊 Smart Analysis Reporter

**Author:** Kavin Ganapathy · ID 100008820
**Course deliverable:** Tools & Methods of Data Analysis

A web-based statistical analysis **and reporting** toolkit. Load a dataset, run a full battery of
descriptive and inferential analyses, and assemble a clean, **editable analytics report** that
exports to **PDF, Word and HTML** — all in the browser, no code required.

The app ships with a default dataset (Kaggle *Student Performance Prediction*, 10,000 rows) so it
**analyses data and produces a report out of the box**.

---

## ✨ Features

| Page | What it does |
|------|--------------|
| **Home** | Auto-loads the dataset, shows an overview, and a one-click **✨ Auto-generate report** |
| **Report Builder** | Editable cover (title, author, summary), per-section editable inferences, reorder/remove, **download PDF / Word / HTML** |
| **Data Profiler** | Shape, missing/duplicate counts, dtype + measurement level (Metric/Ordinal/Nominal) |
| **Descriptive Statistics** | Mean, median, mode, variance, SD, IQR, skewness, kurtosis; histogram + normal curve; box plot |
| **Frequency Tables** | Categorical counts and grouped/binned metric frequencies |
| **Q-Q Plot** | Normality check — Q-Q plot, histogram + normal curve, Shapiro-Wilk test |
| **Correlation & Hypothesis Testing** | Pearson/Spearman/Kendall heatmap; one-sample / Welch / paired t-tests, ANOVA, chi-square, Z-tests — each with **live assumption checks** and effect sizes |
| **Regression** | Simple linear (equation, R², fit plot) and multiple OLS |
| **Time Series** | Trend line + short-horizon forecast (for date-bearing datasets) |

Every analysis page has a **➕ Add to report** button that captures the chart/table plus an
auto-written interpretation you can edit later.

---

## 🚀 Run it locally

```bash
# 1. create a virtual environment
python -m venv venv

# 2. activate it
#   Windows (PowerShell):
venv\Scripts\Activate.ps1
#   macOS / Linux:
source venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt

# 4. launch
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## 🌐 Deploy a public link (Git + Streamlit Community Cloud)

Streamlit Community Cloud is free and hosts directly from a GitHub repo — no server to manage.

### 1. Push the project to GitHub

```bash
git init
git add .
git commit -m "Smart Analysis Reporter"
git branch -M main

# create an EMPTY repo on github.com first, then:
git remote add origin https://github.com/<your-username>/smart-analysis-reporter.git
git push -u origin main
```

> The default dataset is committed at `data/student_dataset.csv`, so the deployed app works
> without any Kaggle credentials.

### 2. Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**.
3. Choose your repo, branch **main**, and main file **`app.py`**.
4. Click **Deploy**. The first build installs `requirements.txt` (a few minutes).
5. You get a public URL like
   `https://<your-app-name>.streamlit.app` — **that is your public link.**

### Alternative one-off public link (no deploy)
From a local run you can expose it temporarily:
```bash
pip install pyngrok
# in another terminal, after `streamlit run app.py`:
python -c "from pyngrok import ngrok; print(ngrok.connect(8501))"
```

---

## 📄 Documentation & deliverables

Generated PDFs live in [`docs/`](docs/):

| File | Contents |
|------|----------|
| **`docs/Analysis_Documentation.pdf`** | A guided statistical study of the dataset built with the report engine — problem statements for (a) descriptive stats, (b) categorical testing, (c) correlation & regression, (d) predictive/linear-regression trends, and (e) a synthesis tying it together. |
| **`docs/System_Documentation.pdf`** | Abstract, system design (with architecture diagram), tech stack, dataset introduction, analysis, conclusion of analysis, and project conclusion — with charts. |

Regenerate them anytime:

```bash
python docs/generate_docs.py
```

---

## 🧪 Example analyses (try these)

1. **Predict exam score** → *Regression → Simple Linear*: X = `study_hours`, Y = `exam_score`.
2. **Does placement relate to scores?** → *Correlation & Hypothesis Testing → Independent t-test*:
   numeric = `exam_score`, grouping = `placement_status`.
3. **Is exam score normal?** → *Q-Q Plot*: variable = `exam_score`.
4. **One-click report** → Home → **✨ Auto-generate report** → **Report Builder** → edit → **Download PDF**.

---

## 📦 Sample dataset

**Student Performance Prediction** (Kaggle: `shambhurajejagadale/student-performance-prediction-dataset`)
— 10,000 students × 8 variables: `study_hours`, `attendance`, `sleep_hours`, `internet_usage`,
`assignments_completed`, `previous_score`, `exam_score`, `placement_status`.
Cached locally at `data/student_dataset.csv`. Upload your own CSV/Excel from the home page anytime.

---

## 🗂️ Project structure

```
analytics_toolkit/
├── app.py                     # home: dataset + auto-report
├── modules/
│   ├── ui.py                  # white/orange theme, header, KPI cards
│   ├── datasets.py            # default dataset loader (cached)
│   ├── data_loader.py         # file loading + column classification
│   ├── descriptive.py         # summary statistics
│   ├── frequency.py           # frequency tables
│   ├── normality.py           # Q-Q + Shapiro-Wilk
│   ├── correlation.py         # correlation matrices
│   ├── regression.py          # simple + multiple OLS
│   ├── tests.py               # t / ANOVA / chi-square / Z + effect sizes
│   ├── assumptions.py         # live test assumption checks
│   ├── timeseries.py          # trend + forecast
│   ├── report.py              # report state, charts, PDF/DOCX/HTML export
│   ├── autoreport.py          # one-click full analysis report
│   └── docgen.py              # Analysis & System documentation PDFs
├── pages/                     # Streamlit multipage UI (1–8)
├── data/student_dataset.csv   # default dataset
├── docs/                      # generated PDFs + generator
└── requirements.txt
```
