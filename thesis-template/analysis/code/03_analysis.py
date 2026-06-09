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

# ── Data preparation ──────────────────────────────────────────────────────────
# Drop rows with missing values in all core analysis variables.
# critical_consumption_mean is NaN for 1 participant (session 1647567)
# who completed fewer than 9 of 11 scale items — excluded per 80% threshold
# applied in 01_build_dataset.py.

df_analysis = df.dropna(
    subset=["perceived_ai_usage", "ai_attitude_mean", "critical_consumption_mean"]
).reset_index(drop=True)

print(
    f"Analysis sample: {df_analysis['session_id'].nunique()} participants, "
    f"{len(df_analysis)} rows"
)
print(
    "Note: 1 participant excluded due to incomplete critical consumption scale "
    "(< 9 of 11 items completed)"
)

# ── Descriptives ──────────────────────────────────────────────────────────────
print(df_analysis.info())
print(df_analysis.isna().sum().sort_values(ascending=False))

desc = df_analysis[
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

corr = df_analysis[
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

# ── Reliability analysis ──────────────────────────────────────────────────────
# Computed in 01_build_dataset.py from raw item scores.
# Results loaded here for reference and reporting.
reliability = pd.read_csv(processed / "reliability.csv")
print("\n=== Reliability Analysis (Cronbach's Alpha) ===")
print(reliability.to_string(index=False))

# ── Scale information ─────────────────────────────────────────────────────────
scale_info = {
    "message_credibility": "1-7 (1=describes very poorly, 7=describes very well)",
    "perceived_ai_usage": "1-7 (1=not at all, 7=entirely)",
    "ai_attitude_mean": "1-7 (1=strongly disagree, 7=strongly agree)",
    "critical_consumption_mean": "1-7 (1=strongly disagree, 7=strongly agree)",
    "ai_disclosure": "0=no disclosure, 1=partial AI, 2=full AI",
}
print("\n=== Scale Information ===")
for var, info in scale_info.items():
    print(f"{var:30s}: {info}")

# ── Per-participant baseline summary ──────────────────────────────────────────
baseline_summary = (
    df_analysis.groupby("session_id")
    .agg(
        mean_credibility=("message_credibility", "mean"),
        sd_credibility=("message_credibility", "std"),
        ai_attitude=("ai_attitude_mean", "first"),
        critical_consumption=("critical_consumption_mean", "first"),
        age=("age", "first"),
        gender=("gender", "first"),
        education=("education", "first"),
    )
    .reset_index()
)

print("=== Per-Participant Baseline Summary ===")
print(baseline_summary.sort_values("mean_credibility"))
baseline_summary.to_csv(processed / "participant_baseline_summary.csv", index=False)

# ── Descriptive: education and profession counts ──────────────────────────────
# Reported descriptively rather than modeled due to sparse cells with N=19
per_person = df_analysis.groupby("session_id")[
    ["education", "profession", "gender"]
].first()

print("\n=== Education Distribution ===")
print(per_person["education"].value_counts())

print("\n=== Profession Distribution ===")
print(per_person["profession"].value_counts())

print("\n=== Gender Distribution ===")
print(per_person["gender"].value_counts())

# ── CMC descriptive check ─────────────────────────────────────────────────────
# Mean perceived AI usage by disclosure level and CMC group
# (preliminary look before Block 5 model)
df_analysis["cmc_group"] = pd.qcut(
    df_analysis["critical_consumption_mean"], q=2, labels=["low_cmc", "high_cmc"]
)

cmc_check = (
    df_analysis.groupby(["ai_disclosure", "cmc_group"], observed=False)[
        "perceived_ai_usage"
    ]
    .agg(["mean", "std", "count"])
    .round(2)
)

print("\n=== Perceived AI Usage by Disclosure × CMC Group ===")
print(cmc_check)

# ── Post summary ──────────────────────────────────────────────────────────────
post_summary = df_analysis.groupby("post_id")[
    ["message_credibility", "perceived_ai_usage", "ai_disclosure"]
].agg(["mean", "std", "count"])
post_summary.to_csv(processed / "post_summary.csv")
print("\n=== Post Summary ===")
print(post_summary)

# ── Block 1: RQ Model ─────────────────────────────────────────────────────────
# Core question: does AI disclosure level affect message credibility?
# Random intercept per participant accounts for repeated measures.
# Continuous covariates only — categorical demographics cause singularity
# with N=19 due to sparse cells (see Block 4 for demographic exploration).

model_rq = smf.mixedlm(
    "message_credibility ~ C(ai_disclosure, Treatment(reference=0)) "
    "+ ai_attitude_mean + critical_consumption_mean + age",
    data=df_analysis,
    groups=df_analysis["session_id"],
).fit(reml=True)

print("\n=== Block 1: RQ Model (Mixed LM, N=19) ===")
print(model_rq.summary())

# ── Block 2a: Disclosure × Critical Consumption ───────────────────────────────
# Does critical media consumption moderate the disclosure → credibility effect?
# Exploratory given small N.

model_sq_a = smf.mixedlm(
    "message_credibility ~ C(ai_disclosure, Treatment(reference=0)) "
    "+ critical_consumption_mean "
    "+ C(ai_disclosure, Treatment(reference=0)):critical_consumption_mean "
    "+ age",
    data=df_analysis,
    groups=df_analysis["session_id"],
).fit(reml=True)

print("\n=== Block 2a: Disclosure × Critical Consumption (Mixed LM) ===")
print(model_sq_a.summary())

# ── Block 2b: Perceived AI Usage × AI Attitude ────────────────────────────────
# Does AI attitude moderate the perceived AI usage → credibility effect?
# Exploratory given small N.

model_sq_b = smf.mixedlm(
    "message_credibility ~ perceived_ai_usage + ai_attitude_mean "
    "+ perceived_ai_usage:ai_attitude_mean + age",
    data=df_analysis,
    groups=df_analysis["session_id"],
).fit(reml=True)

print("\n=== Block 2b: Perceived AI Usage × AI Attitude (Mixed LM) ===")
print(model_sq_b.summary())

# ── Block 3: Mediation ────────────────────────────────────────────────────────
# Chain: ai_disclosure → perceived_ai_usage → message_credibility
#
# Tests whether the effect of disclosure on credibility is carried
# (partly or fully) through changes in perceived AI usage.
#
# ai_disclosure treated as continuous (0/1/2 linear trend) —
# a simplification noted as a limitation.
#
# Note: OLS used here because statsmodels Mediation does not support
# mixed models. Non-independence of observations within participants
# is a limitation of this estimate.

mediator_model = smf.ols(
    "perceived_ai_usage ~ ai_disclosure "
    "+ ai_attitude_mean + critical_consumption_mean + age",
    data=df_analysis,
)

outcome_model = smf.ols(
    "message_credibility ~ perceived_ai_usage + ai_disclosure "
    "+ ai_attitude_mean + critical_consumption_mean + age",
    data=df_analysis,
)

med = Mediation(
    outcome_model,
    mediator_model,
    exposure="ai_disclosure",
    mediator="perceived_ai_usage",
)

med_result = med.fit(n_rep=1000)

print("\n=== Block 3: Mediation Analysis ===")
print(med_result.summary())

# ── Block 4: Participant baseline predictors ──────────────────────────────────
# Do personal characteristics predict overall credibility rating tendency?
# Gender included as the only categorical demographic — education and profession
# have too many sparse categories (1-2 people per level) for reliable estimation
# with N=19. Education and profession are reported descriptively above instead.
# Results interpreted as exploratory.

model_baseline = smf.mixedlm(
    "message_credibility ~ ai_attitude_mean + critical_consumption_mean "
    "+ age + C(gender) "
    "+ C(ai_disclosure, Treatment(reference=0))",
    data=df_analysis,
    groups=df_analysis["session_id"],
).fit(reml=True)

print("\n=== Block 4: Participant Baseline Predictors (Exploratory) ===")
print(model_baseline.summary())

# ── Block 5: CMC moderates disclosure → perceived AI usage ───────────────────
# Core question: do high critical media consumption participants form
# perceived AI usage judgments that diverge from the disclosed level?
# OLS used instead of mixed LM because between-person variance in
# perceived_ai_usage is near zero (Group Var = 0.212 in mixed model),
# which caused convergence failure. HC3 robust standard errors used instead.

model_cmc_mod = smf.ols(
    "perceived_ai_usage ~ C(ai_disclosure, Treatment(reference=0)) "
    "+ critical_consumption_mean "
    "+ C(ai_disclosure, Treatment(reference=0)):critical_consumption_mean "
    "+ age",
    data=df_analysis,
).fit(cov_type="HC3")

print("\n=== Block 5: CMC Moderates Disclosure → Perceived AI Usage (OLS) ===")
print(model_cmc_mod.summary())
