import requests
import pandas as pd
import os
import time
from tqdm import tqdm
from datetime import datetime

debug = False

base_url = "https://lgdc.uml.edu/common/DIDBGetValues"
char_names = ",".join([
    "CS", "foF2", "foF1", "foE", "foEs", "fbEs", "foEa", "foP", "fxI", "MUFD", "MD",
    "hF2", "hF", "hE", "hEs", "hEa", "hP", "TypeEs", "hmF2", "hmF1", "hmE",
    "zhalfNm", "yF2", "yF1", "yE", "scaleF2", "B0", "B1", "D1", "TEC",
    "FF", "FE", "QF", "QE", "fmin", "fminF", "fminE", "fminEs", "foF2p"
])
availability_stats = []

start_year = 2025
end_year = 2025
years = list(range(start_year, end_year + 1))

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "athens_data")
os.makedirs(output_dir, exist_ok=True)

def fetch_month(year, month, max_retries=3, delay=5):
    # Stop fetching if you reached the present month
    today = datetime.now()
    if year == today.year and month > today.month:
        tqdm.write(f"\033[93m{year}-{month:02d} is next month. You need to wait some days and try again, yo.\033[0m")
        return "STOP"

    from_date = f"{year}/{month:02d}/01 00:00:00"
    to_date = f"{year+1}/01/01 00:00:00" if month == 12 else f"{year}/{month+1:02d}/01 00:00:00"
    desc = f"{year}-{month:02d}"

    params = {
        "ursiCode": "AT138",
        "charName": char_names,
        "DMUF": "3000",
        "fromDate": from_date,
        "toDate": to_date,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()

            lines = response.text.splitlines()

            # Check for ERROR lines early
            for line in lines[:100]:
                if line.startswith("ERROR"):
                    tqdm.write(f"\033[91m{desc}: Server returned error: {line}\033[0m")
                    for col in char_names.split(","):
                        availability_stats.append({
                            "year": year,
                            "month": month,
                            "column": col,
                            "availability": 0.0
                        })
                    return "ERROR"

            if not lines:
                raise ValueError("Empty response")

            header_line = next(line for line in lines if line.startswith("#Time"))
            raw_cols = header_line.strip("#").strip().split()
            col_names = [raw_cols[i] for i in range(0, len(raw_cols), 2)]
            data_start = lines.index(header_line) + 1

            data_rows = []
            for line in lines[data_start:]:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.strip().split()
                values = [parts[i] for i in range(0, len(parts), 2)]
                if len(values) != len(col_names):
                    raise ValueError(f"Mismatched columns in line: {line}")
                data_rows.append(values)

            if not data_rows:
                raise ValueError("No valid data rows found")

            df = pd.DataFrame(data_rows, columns=col_names)
            df["Time"] = pd.to_datetime(df["Time"])
            for col in df.columns:
                if col != "Time":
                    df[col] = pd.to_numeric(df[col].replace("---", pd.NA), errors="coerce")

            threshold = 0.95
            nan_percentages = df.isna().mean()
            for col, pct in nan_percentages.items():
                availability = (1 - pct) * 100
                availability_stats.append({
                    "year": year,
                    "month": month,
                    "column": col,
                    "availability": availability
                })
                if pct > threshold and debug:
                    tqdm.write(f"Tossing column {col} of {desc} because {pct * 100:.2f}% NaN")

            df = df.loc[:, df.isna().mean() < threshold]

            output_path = os.path.join(output_dir, str(year), f"{year}_{month:02d}.csv")
            df.to_csv(output_path, index=False)
            if len(df)<100:
                tqdm.write(f"\033[91m{desc}: Saved only ({len(df)} rows)\033[0m")
            else:
                tqdm.write(f"{desc}: {len(df)}")
            return "OK"

        except Exception as e:
            tqdm.write(f"\033[91m{desc}: Attempt {attempt} failed - {e}\033[0m")
            if attempt < max_retries:
                tqdm.write(f"\033[91m{desc}: Retrying in {delay} seconds...\033[0m")
                time.sleep(delay)
                delay += delay/1.5
            else:
                raise SystemExit(f"{desc}: All {max_retries} attempts failed. Aborting.")
    return "END"


# Create yearly subfolders and save monthly data
for year in years:
    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    for month in tqdm(range(1, 13), desc=f"Processing {year} months"):
        result = fetch_month(year, month)
        if result == "STOP":
            tqdm.write(f"\033[90mGracefully stopping month fetching.\033[0m")
            # stop fetching further months
            break

# Convert availability stats to DataFrame
output_path = os.path.join(output_dir, "column_availability.csv")

# Load existing data if it exists
if os.path.exists(output_path):
    existing_df = pd.read_csv(output_path)
    # Drop any overlapping entries (same year, month, column)
    existing_df = existing_df[~existing_df.set_index(['year', 'month', 'column']).index.isin(
        pd.DataFrame(availability_stats).set_index(['year', 'month', 'column']).index
    )]
    # Combine old (filtered) + new
    availability_df = pd.concat([existing_df, pd.DataFrame(availability_stats)], ignore_index=True)
else:
    availability_df = pd.DataFrame(availability_stats)
availability_df.to_csv(output_path, index=False)

for year, group in availability_df.groupby("year"):
    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    yearly_mean = group.groupby("column")["availability"].mean().sort_values(ascending=False)
    output_path = os.path.join(year_dir, "column_availability_mean.csv")
    yearly_mean.to_csv(output_path, header=["mean_availability"])

import matplotlib.pyplot as plt

for year, group in availability_df.groupby("year"):
    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    yearly_pivot = group.pivot(index="month", columns="column", values="availability")
    top_columns = yearly_pivot.mean().sort_values(ascending=False).index

    plt.figure(figsize=(17, 10))
    handles = []
    labels = []

    for i, col in enumerate(top_columns):
        linestyle = ['-', '--', '-.', ':'][i % 4]
        line, = plt.plot(yearly_pivot.index, yearly_pivot[col], marker='o', linestyle=linestyle, label=col, alpha=0.8)
        handles.append(line)
        labels.append(col)

    availability_order = yearly_pivot[top_columns].mean().sort_values(ascending=False)
    sorted_labels = availability_order.index.tolist()
    sorted_handles = [handles[labels.index(lbl)] for lbl in sorted_labels]

    plt.title(f"Monthly Data Availability by Column - {year}")
    plt.ylabel("Availability (%)")
    plt.xlabel("Month")
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    yearly_plot_path = os.path.join(year_dir, "column_availability.png")
    plt.savefig(yearly_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
