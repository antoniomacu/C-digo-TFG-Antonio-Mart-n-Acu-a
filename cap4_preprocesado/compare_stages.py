import pandas as pd

s1 = pd.read_csv("htf_pumps_annotations_stage1.csv")
s2 = pd.read_csv("htf_pumps_annotations.csv")

# Merge on the natural period key
key = ["date", "pump", "period_start"]
merged = s1[key + ["classification", "reasons"]].merge(
    s2[key + ["classification", "reasons"]],
    on=key,
    suffixes=("_s1", "_s2"),
)

# Periods that changed normal → abnormal (ensemble's net reclassifications)
changed = merged[
    (merged["classification_s1"] == "normal") &
    (merged["classification_s2"] == "abnormal")
].copy()

print(f"\nTotal periods:            {len(merged)}")
print(f"Changed normal→abnormal:  {len(changed)}")
print(f"Unchanged:                {len(merged) - len(changed)}")

print("\nPer-pump breakdown:")
for pump_id in sorted(changed["pump"].unique()):
    n = len(changed[changed["pump"] == pump_id])
    print(f"  Pump {pump_id}: {n} periods reclassified")

print("\nReclassified pump-days (date, pump, deciding reasons):")
print(changed[["date", "pump", "period_start", "reasons_s2"]].to_string(index=False))

changed.to_csv("stage1_vs_stage2_diff.csv", index=False)
print("\nFull diff saved to stage1_vs_stage2_diff.csv")
