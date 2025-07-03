import csv
import os
import requests
import re
from geopy.distance import geodesic

script_dir = os.path.dirname(os.path.abspath(__file__))
kalpaki_location = (39.9126, 20.5888)
parnon_location = (37.2609, 22.5847)

def distance_between_locations(coil_location, earthquake_location=(37.7749, -122.4194)):
    distance = geodesic(coil_location, earthquake_location).kilometers
    print(f"Distance: {distance:.2f} km")
    return distance

output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Map for month conversion
month_map = {
    'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
    'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
    'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
}

def convert_date_format(date_str):
    parts = date_str.strip().split()
    if len(parts) == 3:
        year = parts[0]
        month = month_map.get(parts[1].upper(), '01')
        day = parts[2].zfill(2)
        return f"{day}_{month}_{year}"
    return "unknown"

def parse_date_key(date_str):
    try:
        d, m, y = map(int, date_str.split("_"))
        return y * 10000 + m * 100 + d
    except:
        return -1

def clean_old_files(dir, start_key, end_key, start_str, ext):
    for fname in os.listdir(dir):
        if fname.endswith(ext) and fname.startswith(start_str):
            match = re.match(rf"^{start_str}_to_(\d+_\d+_\d+|unknown)\.{ext}$", fname)
            if match:
                end_part = match.group(1)
                end_candidate_key = parse_date_key(end_part)
                if end_candidate_key == -1 or end_candidate_key < end_key:
                    os.remove(os.path.join(dir, fname))
                    print(f"🗑️ Removed outdated file: {fname}")

def process_and_save_files():
    # Download data
    url = "https://bbnet2.gein.noa.gr/current_catalogue/current_catalogue_year2.php"
    response = requests.get(url)
    lines = response.text.strip().splitlines()

    headers = lines[0].strip().split()
    metadata = lines[1].strip().split()

    full_headers = []
    for i in range(len(headers)):
        header = headers[i]
        meta = metadata[i - 1] if i > 0 and (i - 1) < len(metadata) else ""
        full_headers.append(f"{header} {meta}".strip())

    data = []
    raw_dates = []
    for line in lines[2:]:
        date_raw = line[:13].strip()
        parts = [
            date_raw,
            line[13:26].strip(),
            line[26:34].strip(),
            line[34:42].strip(),
            line[42:50].strip(),
            line[50:].strip(),
        ]
        if all(p and p.lower() != "nan" for p in parts):
            raw_dates.append(date_raw)
            data.append(parts)

    start_date_fmt = convert_date_format(raw_dates[0]) if raw_dates else "unknown_start"
    end_date_fmt = convert_date_format(raw_dates[-1]) if raw_dates else "unknown_end"
    start_key = parse_date_key(start_date_fmt)
    end_key = parse_date_key(end_date_fmt)

    base_name = f"{start_date_fmt}_to_{end_date_fmt}"
    output_csv = os.path.join(output_dir, base_name + ".csv")
    output_txt = os.path.join(script_dir, base_name + ".txt")

    # Clean old matching CSVs and TXTs
    clean_old_files(output_dir, start_key, end_key, start_date_fmt, "csv")
    clean_old_files(script_dir, start_key, end_key, start_date_fmt, "txt")

    # Save TXT
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Save CSV
    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(full_headers)
        writer.writerows(data)

    csv_row_count = sum(1 for _ in open(output_csv)) - 1
    txt_row_count = len(data)

    if csv_row_count == txt_row_count:
        print(f"✅ Validation passed: {output_txt} -> {output_csv}")
    else:
        print(f"❌ Validation failed: {output_txt} -> {output_csv}")
        print(f"   Rows in TXT: {txt_row_count}, Rows in CSV: {csv_row_count}")

process_and_save_files()
print("All files processed.")
