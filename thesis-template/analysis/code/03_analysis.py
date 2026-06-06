from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.mediation import Mediation

processed = Path(
    "/Users/juliaseibold/Desktop/SoSe 26/Seminar/seminar_thesis/thesis-template/analysis/processed"
)

df = pd.read_csv(processed / "long_participant_data.csv")

for c in [
    "age",
    "ai_attitude_mean",
    "critical_consumption_mean",
    "message_credibility",
    "perceived_ai_usage",
    "ai_disclosure",
    "post_id",
]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ── Descriptives ────────────────────────────────────────────────────────────
print(df.info())
print(df.isna().sum().sort_values(ascending=False))

desc = df[
    [
        "message_credibility",
        "perceived_ai_usage",
        "ai_disclosure",
        "ai_attitude_mean",
        "critical_consumption_mean",
        "age",
    ]
].describe()
print(desc)

corr = df[
    [
        "message_credibility",
        "perceived_ai_usage",
        "ai_disclosure",
        "ai_attitude_mean",
        "critical_consumption_mean",
        "age",
    ]
].corr()
print(corr)

# ── Block 1: RQ Model ────────────────────────────────────────────────────────
# Does disclosure level affect perceived message credibility?
# Random intercept per participant handles repeated measures.

model_rq = smf.mixedlm(
    "message_credibility ~ C(ai_disclosure, Treatment(reference=0)) "
    "+ ai_attitude_mean + critical_consumption_mean + age + C(post_id)",
    data=df,
    groups=df["session_id"],
).fit(reml=True)

print("=== Block 1: RQ Model (Mixed LM) ===")
print(model_rq.summary())

# ── Block 2: SQ Model ────────────────────────────────────────────────────────
# Do personal factors moderate the disclosure → credibility relationship?
# Treat as exploratory given N=15.

model_sq = smf.mixedlm(
    "message_credibility ~ C(ai_disclosure, Treatment(reference=0)) "
    "+ ai_attitude_mean + critical_consumption_mean + age "
    "+ C(ai_disclosure, Treatment(reference=0)):critical_consumption_mean "
    "+ perceived_ai_usage:ai_attitude_mean",
    data=df,
    groups=df["session_id"],
).fit(reml=True)

print("=== Block 2: SQ Model — Exploratory Interactions (Mixed LM) ===")
print(model_sq.summary())

# ── Block 3: Mediation ───────────────────────────────────────────────────────
# Chain: ai_disclosure → perceived_ai_usage → message_credibility
#
# ai_disclosure is 0/1/2 — we treat it as continuous here (linear trend)
# which is a simplification but reasonable for a seminar paper.
#
# Mediator model:  perceived_ai_usage ~ ai_disclosure  (+ covariates)
# Outcome model:   message_credibility ~ perceived_ai_usage + ai_disclosure (+ covariates)
#
# The indirect effect (a*b) captures how much of the disclosure effect
# on credibility runs through perceived AI usage.

covariates = "+ ai_attitude_mean + critical_consumption_mean + age + C(post_id)"

mediator_model = smf.ols(f"perceived_ai_usage ~ ai_disclosure {covariates}", data=df)

outcome_model = smf.ols(
    f"message_credibility ~ perceived_ai_usage + ai_disclosure {covariates}", data=df
)

med = Mediation(
    outcome_model,
    mediator_model,
    exposure="ai_disclosure",
    mediator="perceived_ai_usage",
)

med_result = med.fit(n_rep=1000, seed=42)  # bootstrap with 1000 replications

print("=== Block 3: Mediation Analysis ===")
print(med_result.summary())
