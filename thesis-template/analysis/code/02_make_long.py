from pathlib import Path

import pandas as pd

processed = Path(
    "/Users/juliaseibold/Desktop/SoSe 26/Seminar/seminar_thesis/thesis-template/analysis/processed"
)
wide = pd.read_csv(processed / "combined_participant_data.csv")

rows = []
for _, r in wide.iterrows():
    for post_id in range(1, 8):
        rows.append(
            {
                "session_id": r["session_id"],
                "trial_nr": r["trial_nr"],
                "condition_id": r["condition_id"],
                "age": r["age"],
                "gender": r["gender"],
                "education": r["education"],
                "profession": r["profession"],
                "ai_attitude_mean": r["ai_attitude_mean"],
                "critical_consumption_mean": r["critical_consumption_mean"],
                "post_id": post_id,
                "message_credibility": r[f"post{post_id}_credibility_mean"],
                "perceived_ai_usage": r[f"post{post_id}_ai_usage"],
                "ai_disclosure": r[f"post{post_id}_disclosure"],
                "ai_disclosure_label": r[f"post{post_id}_disclosure_label"],
            }
        )

long_df = pd.DataFrame(rows)
long_df.to_csv(processed / "long_participant_data.csv", index=False)

print(long_df.head())
print("Rows:", len(long_df))
