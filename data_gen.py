import numpy as np
import pandas as pd
from pathlib import Path


def generate_patient_vitals(n: int = 250, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 90, size=n)
    sex = rng.choice(["M", "F"], size=n)

    bmi = rng.normal(26, 4, size=n).clip(15, 50)
    extreme_idx = rng.choice(n, size=12, replace=False)
    bmi[extreme_idx] = rng.uniform(40, 48, size=12)

    hba1c = 5.0 + (bmi - 22) * 0.08 + rng.normal(0, 0.4, size=n)
    hba1c = hba1c.clip(4.0, 12.0)
    diabetic_idx = rng.choice(n, size=5, replace=False)
    hba1c[diabetic_idx] = rng.uniform(8.0, 11.0, size=5)

    systolic = 110 + (age - 40) * 0.3 + rng.normal(0, 12, size=n)
    systolic = systolic.clip(85, 200)
    htn_idx = rng.choice(n, size=8, replace=False)
    systolic[htn_idx] = rng.uniform(170, 195, size=8)

    diastolic = 70 + (systolic - 120) * 0.4 + rng.normal(0, 6, size=n)
    diastolic = diastolic.clip(50, 120)

    visit_date = pd.date_range("2025-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "age": age,
        "sex": sex,
        "bmi": np.round(bmi, 1),
        "hba1c": np.round(hba1c, 2),
        "systolic_bp": np.round(systolic).astype(int),
        "diastolic_bp": np.round(diastolic).astype(int),
        "visit_date": visit_date,
    })


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "patient_vitals.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_patient_vitals()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
