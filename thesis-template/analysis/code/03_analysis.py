from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.mediation import Mediation

processed = Path(
    "/Users/juliaseibold/Desktop/sose26/seminar/seminar_thesis/thesis-template/analysis/processed"
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

# Dummy variables for disclosure contrasts (reference = no disclosure)
# Used in separate mediation models per contrast (Blocks 3a and 3b)
df_analysis["disc_partial"] = (df_analysis["ai_disclosure"] == 1).astype(int)
df_analysis["disc_full"] = (df_analysis["ai_disclosure"] == 2).astype(int)

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
# preliminary look before Block 5 model
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


no_disc = df_analysis[df_analysis["ai_disclosure"] == 0]["perceived_ai_usage"]

t_stat, p_val = stats.ttest_1samp(no_disc, popmean=4.0)
print(
    f"No disclosure perceived AI usage: M = {no_disc.mean():.2f}, SD = {no_disc.std():.2f}"
)
print(f"One-sample t-test vs midpoint 4.0: t = {t_stat:.3f}, p = {p_val:.3f}")

# ── Perceived AI usage by disclosure level ────────────────────────────────────
# Descriptive summary and one-sample t-tests vs scale midpoint (4.0)
# for each disclosure condition separately.

from scipy import stats

print("\n=== Perceived AI Usage by Disclosure Level ===")

disc_labels = {0: "No disclosure", 1: "Partial disclosure", 2: "Full disclosure"}

for level, label in disc_labels.items():
    subset = df_analysis[df_analysis["ai_disclosure"] == level]["perceived_ai_usage"]
    t_stat, p_val = stats.ttest_1samp(subset, popmean=4.0)
    print(
        f"{label:20s}: M = {subset.mean():.2f}, SD = {subset.std():.2f}, "
        f"n = {len(subset)}, t = {t_stat:.3f}, p = {p_val:.3f}"
    )

# ── Block 1: Total effect of disclosure on credibility ───────────────────────
# Establishes the total effect before decomposing via mediation.
# Full mediation is expected based on the conceptual model.
# Random intercept per participant accounts for repeated measures.

model_rq = smf.mixedlm(
    "message_credibility ~ C(ai_disclosure, Treatment(reference=0)) "
    "+ ai_attitude_mean + critical_consumption_mean + age",
    data=df_analysis,
    groups=df_analysis["session_id"],
).fit(reml=True)

print("\n=== Block 1: Total Effect — Disclosure on Credibility (Mixed LM, N=19) ===")
print(model_rq.summary())

# ── Path b: Perceived AI usage → message credibility ─────────────────────────
# Direct test of the core relationship in the conceptual model.
# Mixed LM with random intercept for participant accounts for
# repeated measures. No moderators — clean bivariate relationship.

model_path_b = smf.mixedlm(
    "message_credibility ~ perceived_ai_usage",
    data=df_analysis,
    groups=df_analysis["session_id"],
).fit(reml=True)

print("\n=== Path b: Perceived AI Usage → Message Credibility (Mixed LM) ===")
print(model_path_b.summary())

# ── Age and gender as predictors of perceived AI usage ────────────────────────
# Tests whether demographic variables influence the mediator variable.
# Relevant because perceived AI usage is the central mediator in the model.
# Mixed LM with random intercept for participant accounts for repeated measures.

model_demo_pai = smf.mixedlm(
    "perceived_ai_usage ~ age + C(gender) + C(ai_disclosure, Treatment(reference=0))",
    data=df_analysis,
    groups=df_analysis["session_id"],
).fit(reml=True)

print("\n=== Demographics → Perceived AI Usage (Mixed LM) ===")
print(model_demo_pai.summary())

# ── Block 2b: AI attitude moderates perceived AI usage → credibility ──────────
# Second-stage moderation in the conceptual model.
# Does AI attitude change how strongly perceived AI usage affects credibility?
# Exploratory given small N.

# Center ai_attitude_mean so main effects are interpretable at the mean
df_analysis["ai_attitude_centered"] = (
    df_analysis["ai_attitude_mean"] - df_analysis["ai_attitude_mean"].mean()
)

model_sq_b = smf.mixedlm(
    "message_credibility ~ perceived_ai_usage + ai_attitude_centered "
    "+ perceived_ai_usage:ai_attitude_centered",
    data=df_analysis,
    groups=df_analysis["session_id"],
).fit(reml=True)

print(
    "\n=== Block 2b: AI Attitude Moderates Perceived AI Usage → Credibility (Mixed LM) ==="
)
print(model_sq_b.summary())

# ── Block 3a: Mediation — partial vs no disclosure ───────────────────────────
# Path: partial disclosure → perceived AI usage → credibility
# Tests whether the mediation chain holds for partial disclosure specifically.
# OLS used — statsmodels Mediation does not support mixed models.
# Non-independence of observations within participants is a noted limitation.

mediator_partial = smf.ols(
    "perceived_ai_usage ~ disc_partial "
    "+ ai_attitude_mean + critical_consumption_mean + age",
    data=df_analysis,
)
outcome_partial = smf.ols(
    "message_credibility ~ perceived_ai_usage + disc_partial "
    "+ ai_attitude_mean + critical_consumption_mean + age",
    data=df_analysis,
)
med_partial = Mediation(
    outcome_partial,
    mediator_partial,
    exposure="disc_partial",
    mediator="perceived_ai_usage",
).fit(n_rep=1000)

print("\n=== Block 3a: Mediation — Partial vs No Disclosure ===")
print(med_partial.summary())

# ── Block 3b: Mediation — full vs no disclosure ───────────────────────────────
# Path: full disclosure → perceived AI usage → credibility
# Tests whether the mediation chain holds for full disclosure specifically.

mediator_full = smf.ols(
    "perceived_ai_usage ~ disc_full "
    "+ ai_attitude_mean + critical_consumption_mean + age",
    data=df_analysis,
)
outcome_full = smf.ols(
    "message_credibility ~ perceived_ai_usage + disc_full "
    "+ ai_attitude_mean + critical_consumption_mean + age",
    data=df_analysis,
)
med_full = Mediation(
    outcome_full,
    mediator_full,
    exposure="disc_full",
    mediator="perceived_ai_usage",
).fit(n_rep=1000)

print("\n=== Block 3b: Mediation — Full vs No Disclosure ===")
print(med_full.summary())

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
# First-stage moderation in the conceptual model.
# Does CML change how strongly disclosure shifts perceived AI usage?
# OLS used — near-zero between-person variance in perceived_ai_usage
# caused mixed LM convergence failure (Group Var = 0.212).
# HC3 robust standard errors used instead.

model_cmc_mod = smf.ols(
    "perceived_ai_usage ~ C(ai_disclosure, Treatment(reference=0)) "
    "+ critical_consumption_mean "
    "+ C(ai_disclosure, Treatment(reference=0)):critical_consumption_mean "
    "+ age",
    data=df_analysis,
).fit(cov_type="HC3")

print("\n=== Block 5: CML Moderates Disclosure → Perceived AI Usage (OLS) ===")
print(model_cmc_mod.summary())
