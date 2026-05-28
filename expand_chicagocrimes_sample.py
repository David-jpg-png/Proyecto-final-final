import csv
import datetime
import hashlib
import math
import random

INPUT_CSV = 'chicagocrimes.csv'
OUTPUT_CSV = 'chicagocrimes_sample.csv'
SAMPLE_ROWS = 1000
DERIVED_COLUMNS = 1978
RANDOM_SEED = 42

ORIGINAL_COLUMNS = [
    'ID', 'Case Number', 'Date', 'Block', 'IUCR', 'Primary Type', 'Description',
    'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward',
    'Community Area', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Year',
    'Updated On', 'Latitude', 'Longitude', 'Location'
]


def stable_hash_number(value: str, scale: float = 1000.0) -> float:
    if value is None:
        return 0.0
    value = str(value)
    digest = hashlib.sha256(value.encode('utf-8')).digest()
    h = int.from_bytes(digest[:8], 'little', signed=False)
    return (h % 1000000) / scale


def parse_datetime(value: str):
    if not value:
        return None
    for fmt in ('%m/%d/%Y %I:%M:%S %p', '%m/%d/%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def safe_float(value, default=0.0):
    if value is None or value == '':
        return default
    try:
        return float(value)
    except ValueError:
        return default


def boolean_float(value):
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ('true', '1', 'yes', 'y'): return 1.0
        if v in ('false', '0', 'no', 'n'): return 0.0
    return 0.0


def build_base_features(row):
    dt = parse_datetime(row[2])
    year = safe_float(row[17])
    month = dt.month if dt else safe_float(row[2].split('/')[0]) if '/' in row[2] else 0.0
    day = dt.day if dt else 0.0
    hour = dt.hour if dt else 0.0
    minute = dt.minute if dt else 0.0
    weekday = dt.weekday() if dt else 0.0
    day_of_year = dt.timetuple().tm_yday if dt else 0.0
    quarter = math.ceil(month / 3.0) if month else 0.0
    arrest = boolean_float(row[8])
    domestic = boolean_float(row[9])
    beat = safe_float(row[10])
    district = safe_float(row[11])
    ward = safe_float(row[12])
    community = safe_float(row[13])
    xcoord = safe_float(row[15])
    ycoord = safe_float(row[16])
    latitude = safe_float(row[19])
    longitude = safe_float(row[20])
    center_lat, center_lon = 41.8781, -87.6298
    distance_center = math.hypot(latitude - center_lat, longitude - center_lon)
    hour_sin = math.sin(hour / 24.0 * 2.0 * math.pi)
    hour_cos = math.cos(hour / 24.0 * 2.0 * math.pi)
    block_hash = stable_hash_number(row[3], scale=5000.0)
    case_hash = stable_hash_number(row[1], scale=5000.0)
    primary_hash = stable_hash_number(row[5], scale=1000.0)
    description_hash = stable_hash_number(row[6], scale=1000.0)
    location_hash = stable_hash_number(row[7], scale=1000.0)
    iucr_hash = stable_hash_number(row[4], scale=1000.0)
    fbi_hash = stable_hash_number(row[14], scale=1000.0)

    return [
        year, month, day, hour, minute, weekday, day_of_year, quarter,
        arrest, domestic, beat, district, ward, community,
        xcoord, ycoord, latitude, longitude, distance_center,
        hour_sin, hour_cos, block_hash, case_hash,
        primary_hash, description_hash, location_hash,
        iucr_hash, fbi_hash
    ]


def build_derived_features(base_features, row_index):
    values = []
    n_base = len(base_features)
    rng = random.Random(RANDOM_SEED + row_index)
    for idx in range(DERIVED_COLUMNS):
        a = base_features[idx % n_base]
        b = base_features[(idx * 3 + 1) % n_base]
        c = base_features[(idx * 5 + 2) % n_base]
        weight_a = rng.uniform(-3.0, 3.0)
        weight_b = rng.uniform(-2.0, 2.0)
        weight_c = rng.uniform(-1.0, 1.0)
        bias = rng.uniform(-10.0, 10.0)
        combined = a * weight_a + b * weight_b - c * weight_c + bias
        mode = idx % 7
        if mode == 0:
            value = math.sin(combined)
        elif mode == 1:
            value = math.cos(combined)
        elif mode == 2:
            value = math.tanh(combined / (1.0 + abs(combined)))
        elif mode == 3:
            value = math.log1p(abs(combined)) * math.copysign(1.0, combined)
        elif mode == 4:
            value = (combined ** 2) / (1.0 + abs(combined))
        elif mode == 5:
            value = math.sqrt(abs(combined) + 1.0) * math.copysign(1.0, combined)
        else:
            value = (a * b * 0.0001) + (c * 0.001) + rng.uniform(-0.5, 0.5)
        values.append(value)
    return values


def reservoir_sample_rows(path, sample_size, seed):
    rng = random.Random(seed)
    header = None
    reservoir = []
    with open(path, newline='', encoding='utf-8', errors='ignore') as input_file:
        reader = csv.reader(input_file)
        header = next(reader)
        for i, row in enumerate(reader, start=1):
            if i <= sample_size:
                reservoir.append(row)
            else:
                j = rng.randrange(i)
                if j < sample_size:
                    reservoir[j] = row
    return header, reservoir


def generate_sample():
    print(f'Reading {SAMPLE_ROWS} rows from {INPUT_CSV} with reservoir sampling...')
    header, rows = reservoir_sample_rows(INPUT_CSV, SAMPLE_ROWS, RANDOM_SEED)
    print(f'Creating output file {OUTPUT_CSV} with {len(rows)} rows and {len(ORIGINAL_COLUMNS) + DERIVED_COLUMNS} columns...')

    derived_header = [f'feature_{i+1:04d}' for i in range(DERIVED_COLUMNS)]
    output_header = ORIGINAL_COLUMNS + derived_header

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(output_header)
        for row_index, row in enumerate(rows):
            base_features = build_base_features(row)
            derived = build_derived_features(base_features, row_index)
            writer.writerow(row + [f'{v:.6f}' for v in derived])

    print('Sample dataset generation complete.')


if __name__ == '__main__':
    generate_sample()
