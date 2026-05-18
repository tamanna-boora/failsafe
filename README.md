# FAILSAFE

Predicts which students are likely to fail a semester — and tells you why, so faculty can step in before the final results come out.

---

## the problem

So here's the thing — by the time end-sem results show who failed, it's already too late to actually help anyone. Most colleges have no early warning system. A student could be struggling for months and nobody notices until grades drop.

I wanted to fix that. Or at least take a shot at it.

FAILSAFE looks at things like attendance, study time, family background, past grades — basically all the stuff that quietly predicts trouble — and flags students who are heading toward failure. Then it explains why (using SHAP) and suggests what faculty could actually do about it.

---

## what's inside

- Risk prediction per student (uses XGBoost, not just final grades)
- SHAP waterfall plots that show which factors pushed each student into "at-risk"
- Auto-generated intervention list per student — things like "study skills workshop", "weekly check-in", "counsellor referral"
- A dashboard with filters, summary charts, and per-student deep dives

---

## screenshots

![Dashboard overview](screenshots01-dashboard.png)

![Predictions overview](screenshots02-predictions.png)

![SHAP waterfall](screenshots03-shap.png)

![Interventions tab](screenshots04-interventions.png)


---

## stack

- XGBoost for the model
- SHAP for the "why"
- Streamlit for the UI (originally planned React + FastAPI but the deadline was 4 days, so… Streamlit)
- UCI Student Performance dataset (649 students, Portuguese schools)

---

## numbers

- Accuracy: 89.23%
- F1: 0.6957
- Confusion matrix: TP=16, FP=10, FN=4, TN=100

Tuned more for recall than precision — false alarms are way better than missing a kid who actually needs help.

---

## run it

```bash
git clone https://github.com/tamanna-boora/failsafe.git
cd failsafe

python -m venv venv
source venv/Scripts/activate     # Windows / Git Bash
# source venv/bin/activate        # Mac / Linux

pip install -r requirements.txt
python download_data.py
python src/train_model.py
streamlit run app/app.py
```

Opens at `localhost:8501`.

---

## folders
failsafe/
├── app/app.py              # the streamlit dashboard
├── src/train_model.py      # ML pipeline
├── src/interventions.py    # rule-based intervention generator
├── data/students.csv       # downloaded by download_data.py
├── models/                 # trained model + encoders go here
├── LEARNING.md             # what I learned (reflection)
├── NOTES.md                # dev notes from while I was building
└── README.md

---

## decisions I made

- **XGBoost** instead of logistic regression — handles mixed categorical + numeric without much fuss, and gives feature importance for free
- **Streamlit** instead of React — the spec said React but with 4 days solo, no chance. Streamlit got me a real app fast
- **Rule-based interventions** — easier to explain to faculty than something the model spits out, and I'd need way more labelled data to train a real intervention model
- **Top 20 at-risk shown in deep dive** — 100+ student cards on one page was unusable

---

## what could be better

- Dataset is from Portuguese schools, not Indian colleges. Patterns probably generalise but specifics won't.
- G1 and G2 (first/second period grades) dominate the predictions, which makes the model strong but reduces the "early warning" angle — by the time you have G1, the semester is already half over
- Interventions are hardcoded rules, not ML
- Tested on 649 students. More data = more trust

---

## future stuff

- React + FastAPI version (matches the original spec)
- Real IITG data: CGPA, mess attendance, club activity
- Auto-email intervention plans to faculty
---

## try it live

🔗 **[https://failsafe-tamanna.streamlit.app](https://failsafe-tamanna.streamlit.app)**

No setup needed — just click the link and try it.

---

**Demo Video**: https://drive.google.com/file/d/1S4ioHhg6pd91vFkBb3bTU6UB8PkF7GKi/view?usp=sharing

**Report PDF**: https://drive.google.com/file/d/12w7dXDYUOfL1d0mDOD6FK651x_cgq76O/view?usp=drive_link

 FAILSAFE — IIT Guwahati — Coding Club Even Semester Project’26
