import pandas as pd
import numpy as np

# Create a proper sample dataset
df = pd.DataFrame({
    'Date': pd.date_range('2021-01-01', periods=1000, freq='D'),
    'ID': [str(i) for i in range(1, 1001)],
    'Case Number': [f'CA{i:04d}' for i in range(1, 1001)],
    'Location Description': np.random.choice(['STREET', 'RESIDENCE', 'APARTMENT', 'ALLEY', 'PARKING LOT'], 1000),
    'Primary Type': np.random.choice(['THEFT', 'ROBBERY', 'BURGLARY', 'ASSAULT', 'BATTERY'], 1000),
    'Description': np.random.choice(['OVER 500', 'UNDER 500', 'WITH FORCE'], 1000),
    'Arrest': np.random.choice([True, False], 1000, p=[0.35, 0.65]),
    'Domestic': np.random.choice([True, False], 1000, p=[0.2, 0.8]),
    'X Coordinate': np.random.uniform(1168000, 1180000, 1000),
    'Y Coordinate': np.random.uniform(1895000, 1910000, 1000),
    'Latitude': np.random.uniform(41.8, 42.0, 1000),
    'Longitude': np.random.uniform(-87.7, -87.5, 1000),
    'Ward': np.random.randint(1, 51, 1000).astype(str),
    'Community Area': np.random.randint(1, 78, 1000).astype(str),
    'District': np.random.randint(1, 30, 1000).astype(str),
    'FBI Code': np.random.choice(['06', '11', '14'], 1000)
})

df.to_csv('chicagocrimes_sample.csv', index=False)
print('Sample dataset created successfully')
