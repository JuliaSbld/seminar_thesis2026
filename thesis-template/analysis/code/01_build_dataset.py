from pathlib import Path

import numpy as np
import pandas as pd

root = Path(
    "/Users/juliaseibold/Desktop/SoSe 26/Seminar/seminar_thesis/thesis-template/analysis/data"
)
processed = root.parent / "processed"
processed.mkdir(exist_ok=True)


def clean_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def to_numeric_cols(df):
    for c in df.columns:
        if c.startswith(("AI", "MC", "CC", "Age")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def first_nonmissing(s):
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan


def process_trial_file(file_path):
    df = pd.read_csv(file_path).replace("", np.nan)
    df = clean_columns(df)
    df = df.rename(columns={"AI71": "AI70"})
    df = to_numeric_cols(df)

    posts = df[df["Task_Name"] == "Posts"].copy()
    cc = df[df["Task_Name"] == "CriticalConsumption"].copy()
    att = df[df["Task_Name"] == "AI_Attitude"].copy()
    demo = df[df["Task_Name"] == "final_questionnaire"].copy()

    out = {
        "session_file": file_path.as_posix(),
        "session_id": first_nonmissing(df["Rec_Session_Id"])
        if "Rec_Session_Id" in df.columns
        else np.nan,
        "trial_nr": first_nonmissing(df["Trial_Nr"])
        if "Trial_Nr" in df.columns
        else np.nan,
        "condition_id": first_nonmissing(df["Condition_Id"])
        if "Condition_Id" in df.columns
        else np.nan,
        "age": first_nonmissing(demo["Age1"]) if "Age1" in demo.columns else np.nan,
        "gender": first_nonmissing(demo["Gender"])
        if "Gender" in demo.columns
        else np.nan,
        "education": first_nonmissing(demo["Education"])
        if "Education" in demo.columns
        else np.nan,
        "profession": first_nonmissing(demo["Profession"])
        if "Profession" in demo.columns
        else np.nan,
        "attention_check": first_nonmissing(posts["AttentionCheck"])
        if "AttentionCheck" in posts.columns
        else np.nan,
    }

    att_cols = ["AI1", "AI2", "AI3", "AI4"]
    if all(c in att.columns for c in att_cols):
        att["ai_attitude_mean"] = att[att_cols].mean(axis=1)
        out["ai_attitude_mean"] = first_nonmissing(att["ai_attitude_mean"])

    # Critical consumption — require at least 9 of 11 items (80% threshold)
    # to be non-missing for a valid composite mean.
    # Participants with fewer valid responses receive NaN.
    cc_cols = [f"CC{i}" for i in range(10, 21) if f"CC{i}" in cc.columns]
    if cc_cols:
        cc["cc_valid_count"] = cc[cc_cols].notna().sum(axis=1)
        cc["critical_consumption_mean"] = cc[cc_cols].mean(axis=1)
        valid_cc = cc[cc["cc_valid_count"] >= 9]
        out["critical_consumption_mean"] = (
            first_nonmissing(valid_cc["critical_consumption_mean"])
            if not valid_cc.empty
            else np.nan
        )

    post_map = {
        "post1": {"mc": ["MC10", "MC11", "MC12"], "ai": "AI10", "disc": 0},
        "post2": {"mc": ["MC20", "MC21", "MC22"], "ai": "AI20", "disc": 1},
        "post3": {"mc": ["MC30", "MC31", "MC32"], "ai": "AI30", "disc": 1},
        "post4": {"mc": ["MC40", "MC41", "MC42"], "ai": "AI40", "disc": 2},
        "post5": {"mc": ["MC50", "MC51", "MC52"], "ai": "AI50", "disc": 0},
        "post6": {"mc": ["MC60", "MC61", "MC62"], "ai": "AI60", "disc": 2},
        "post7": {"mc": ["MC70", "MC71", "MC72"], "ai": "AI70", "disc": 0},
    }

    disc_label = {0: "no_disclosure", 1: "partial_ai", 2: "full_ai"}

    for post_name, cfg in post_map.items():
        mc_cols = [c for c in cfg["mc"] if c in posts.columns]
        if mc_cols:
            posts[f"{post_name}_credibility_mean"] = posts[mc_cols].mean(axis=1)
            out[f"{post_name}_credibility_mean"] = first_nonmissing(
                posts[f"{post_name}_credibility_mean"]
            )
        if cfg["ai"] in posts.columns:
            out[f"{post_name}_ai_usage"] = first_nonmissing(posts[cfg["ai"]])

        out[f"{post_name}_disclosure"] = cfg["disc"]
        out[f"{post_name}_disclosure_label"] = disc_label[cfg["disc"]]

    return out


# ── Reliability analysis ──────────────────────────────────────────────────────
# Computed here because raw item scores are only available in the trials.csv
# files — they are not passed through to combined_participant_data.csv.
# The same 80% validity threshold applies: participants with fewer than 9
# valid CC items are excluded from the alpha calculation.


def cronbach_alpha(df_items):
    df_items = df_items.apply(pd.to_numeric, errors="coerce").dropna()
    n = df_items.shape[1]
    if n < 2 or len(df_items) < 2:
        return np.nan
    item_vars = df_items.var(axis=0, ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return round((n / (n - 1)) * (1 - item_vars.sum() / total_var), 3)


mc_rows, att_rows, cc_rows = [], [], []

for file_path in root.rglob("trials.csv"):
    try:
        raw = pd.read_csv(file_path).replace("", np.nan)
        raw = clean_columns(raw)
        raw = raw.rename(columns={"AI71": "AI70"})
        raw = to_numeric_cols(raw)

        posts = raw[raw["Task_Name"] == "Posts"].copy()
        att = raw[raw["Task_Name"] == "AI_Attitude"].copy()
        cc = raw[raw["Task_Name"] == "CriticalConsumption"].copy()

        # Message credibility — 3 items per post × 7 posts
        for post_id in range(1, 8):
            cols = [f"MC{post_id}0", f"MC{post_id}1", f"MC{post_id}2"]
            if all(c in posts.columns for c in cols):
                row = posts[cols].dropna()
                if not row.empty:
                    mc_rows.append(
                        row.rename(
                            columns={
                                cols[0]: "item1",
                                cols[1]: "item2",
                                cols[2]: "item3",
                            }
                        ).iloc[0]
                    )

        # AI attitude — AI1 to AI4
        att_cols_list = ["AI1", "AI2", "AI3", "AI4"]
        if all(c in att.columns for c in att_cols_list):
            row = att[att_cols_list].dropna()
            if not row.empty:
                att_rows.append(row.iloc[0])

        # Critical consumption — CC10 to CC20
        # Apply 80% threshold: require at least 9 of 11 items non-missing
        cc_cols_list = [f"CC{i}" for i in range(10, 21)]
        available_cc = [c for c in cc_cols_list if c in cc.columns]
        if len(available_cc) >= 2:
            row = cc[available_cc].dropna()
            if not row.empty:
                valid_count = row.notna().sum(axis=1).iloc[0]
                if valid_count >= 9:
                    cc_rows.append(row.iloc[0])

    except Exception as e:
        print(f"Reliability error for {file_path}: {e}")

mc_df = pd.DataFrame(mc_rows).apply(pd.to_numeric, errors="coerce")
att_df = pd.DataFrame(att_rows).apply(pd.to_numeric, errors="coerce")
cc_df = pd.DataFrame(cc_rows).apply(pd.to_numeric, errors="coerce")

alpha_credibility = cronbach_alpha(mc_df)
alpha_attitude = cronbach_alpha(att_df)
alpha_cc = cronbach_alpha(cc_df)

print("=== Reliability Analysis (Cronbach's Alpha) ===")
print(
    f"Message credibility  (3 items × 7 posts, N={len(mc_df)}): α = {alpha_credibility}"
)
print(
    f"AI attitude          (4 items, N={len(att_df)}):           α = {alpha_attitude}"
)
print(
    f"Critical consumption ({len(cc_df.columns)} items, N={len(cc_df)}): α = {alpha_cc}"
)

reliability = pd.DataFrame(
    {
        "scale": ["message_credibility", "ai_attitude", "critical_consumption"],
        "n_items": [3, 4, len(cc_df.columns) if not cc_df.empty else 0],
        "n_observations": [len(mc_df), len(att_df), len(cc_df)],
        "cronbach_alpha": [alpha_credibility, alpha_attitude, alpha_cc],
    }
)
reliability.to_csv(processed / "reliability.csv", index=False)
print(reliability)


# ── Extract and combine participant data ──────────────────────────────────────
results, failed = [], []

for file_path in root.rglob("trials.csv"):
    try:
        results.append(process_trial_file(file_path))
    except Exception as e:
        failed.append((str(file_path), str(e)))

combined = pd.DataFrame(results)
combined.to_csv(processed / "combined_participant_data.csv", index=False)

pd.DataFrame(failed, columns=["file", "error"]).to_csv(
    processed / "failed_files.csv", index=False
)

print("Processed:", len(combined))
print("Failed:", len(failed))
print(combined.head())
