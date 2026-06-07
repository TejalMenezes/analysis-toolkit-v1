# 🚀 Deployment Guide — Smart Analysis Reporter

This guide has **two separate process flows**:

- **Flow A — Push the project to GitHub**
- **Flow B — Deploy from GitHub to a public link** (Streamlit Community Cloud)

Target repository: **`https://github.com/TejalMenezes/analysis-toolkit-v1`** (branch `main`).

---

## ✅ Pre-flight checklist (already done in this project)

- [x] `requirements.txt` lists every dependency
- [x] `.gitignore` excludes `venv/`, `__pycache__/`, logs, local settings
- [x] Default dataset committed at `data/student_dataset.csv` (so the cloud app needs **no Kaggle login**)
- [x] App entry point is `app.py`
- [x] Git repository initialised and committed locally

---

## 🅰️ Flow A — Push to GitHub

```
┌─────────────┐   git add /     ┌─────────────┐   git push     ┌──────────────┐
│ Local files │ ───commit────▶  │ Local repo  │ ─────────────▶ │   GitHub     │
└─────────────┘                 └─────────────┘                └──────────────┘
```

Run these from the project root (`t:\analytics_toolkit`):

```bash
# 1. point the local repo at your GitHub repo (only needed once)
git remote add origin https://github.com/TejalMenezes/analysis-toolkit-v1.git

# 2. make sure you are on main
git branch -M main

# 3. stage + commit any pending changes
git add .
git commit -m "Smart Analysis Reporter"        # skip if nothing changed

# 4. push
git push -u origin main
```

**If `origin` already exists** (you'll see *"remote origin already exists"*):

```bash
git remote set-url origin https://github.com/TejalMenezes/analysis-toolkit-v1.git
git push -u origin main
```

### Authentication
When prompted for a password, GitHub no longer accepts your account password —
use a **Personal Access Token (PAT)**:

1. GitHub → **Settings → Developer settings → Personal access tokens → Tokens (classic)**.
2. **Generate new token**, tick the **`repo`** scope, copy it.
3. When `git push` asks for a password, **paste the token**.

> Alternatively install the **GitHub CLI** (`gh auth login`) or **GitHub Desktop**, which handle auth for you.

### Verify
Refresh `https://github.com/TejalMenezes/analysis-toolkit-v1` — you should see all the files.

---

## 🅱️ Flow B — Deploy to a public link (Streamlit Community Cloud)

Once the repo is on GitHub, no server setup is needed — Streamlit builds and hosts it for free.

```
┌──────────────┐   connect repo   ┌────────────────────────┐   build+run   ┌───────────────────────┐
│   GitHub     │ ───────────────▶ │ Streamlit Community     │ ────────────▶ │  https://<app>.        │
│  repo (main) │                  │ Cloud (share.streamlit) │               │  streamlit.app  🌐     │
└──────────────┘                  └────────────────────────┘               └───────────────────────┘
```

**Step by step:**

1. Go to **https://share.streamlit.io** and **sign in with GitHub** (authorise access to your repos).
2. Click **“Create app”** → **“Deploy a public app from GitHub.”**
3. Fill the form:
   - **Repository:** `TejalMenezes/analysis-toolkit-v1`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **“Deploy.”**
5. Wait for the first build (it installs `requirements.txt` — usually 2–5 minutes; the log streams live).
6. When it finishes you get your **public URL**, e.g.
   **`https://analysis-toolkit-v1.streamlit.app`** — share that link.

### After deployment
- **Every `git push` to `main` auto-redeploys** the app — no extra steps.
- Manage / reboot / view logs from your dashboard at **https://share.streamlit.io**.
- If the build fails, open the log: it's almost always a missing line in `requirements.txt`.

---

## 🔁 Updating the live app later

```bash
git add .
git commit -m "describe your change"
git push
```
Streamlit Cloud detects the push and rebuilds automatically.

---

## 🆘 Troubleshooting

| Symptom | Fix |
|---------|-----|
| `remote origin already exists` | `git remote set-url origin <url>` then push |
| Push rejected (`non-fast-forward`) | `git pull --rebase origin main` then `git push` |
| Auth fails on push | Use a **Personal Access Token** as the password (see Flow A) |
| Cloud build fails on import | Add the missing package to `requirements.txt`, commit, push |
| App loads but “no dataset” | Confirm `data/student_dataset.csv` is committed (it is, by default) |

---

## ⚡ Quick temporary public link (no deploy, for a demo)

If you just need a link for a few minutes from your own machine:

```bash
# terminal 1
streamlit run app.py
# terminal 2
pip install pyngrok
python -c "from pyngrok import ngrok; print('Public URL:', ngrok.connect(8501))"
```
This link lives only while your machine and the command are running — use **Flow B** for a permanent link.
