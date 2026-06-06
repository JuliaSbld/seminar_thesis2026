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

    cc_cols = [f"CC{i}" for i in range(10, 21) if f"CC{i}" in cc.columns]
    if cc_cols:
        cc["critical_consumption_mean"] = cc[cc_cols].mean(axis=1)
        out["critical_consumption_mean"] = first_nonmissing(
            cc["critical_consumption_mean"]
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
