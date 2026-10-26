import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# LOAD DATASET
# ======================================================

print("\nLoading dataset...\n")

data = pd.read_csv(
    "GlobalLandTemperaturesByCity.csv"
)

print("Original Shape:", data.shape)

print("\nColumns:\n")
print(data.columns)

# ======================================================
# REMOVE MISSING VALUES
# ======================================================

data = data.dropna(
    subset=['AverageTemperature']
)

print("\nAfter removing missing values:")
print(data.shape)

# ======================================================
# FILTER ONLY INDIA
# ======================================================

india_data = data[
    data['Country'] == 'India'
]

print("\nIndia Dataset Shape:")
print(india_data.shape)

# ======================================================
# SHOW AVAILABLE CITIES
# ======================================================

cities = sorted(
    india_data['City'].unique()
)

print("\nAvailable Indian Cities:\n")

print(cities)

# ======================================================
# SELECT CITY
# ======================================================

CITY_NAME = "Chennai"

# Handle old Madras naming
india_data['City'] = (
    india_data['City']
    .astype(str)
    .str.strip()
)

# Some datasets use Madras instead of Chennai
# We include both safely

city_data = india_data[
    india_data['City'].isin(
        ['Madras', 'Chennai']
    )
]

print("\nSelected City Dataset Shape:")
print(city_data.shape)

# ======================================================
# SELECT REQUIRED COLUMNS
# ======================================================

city_data = city_data[
    ['dt', 'AverageTemperature']
]

city_data.columns = [
    'DATE',
    'TAVG'
]

# ======================================================
# DATE CONVERSION
# ======================================================

city_data['DATE'] = pd.to_datetime(
    city_data['DATE'],
    errors='coerce'
)

# Remove invalid dates
city_data = city_data.dropna(
    subset=['DATE']
)

# ======================================================
# SORT DATA
# ======================================================

city_data = city_data.sort_values(
    'DATE'
)

# ======================================================
# REMOVE EXTREME OUTLIERS
# ======================================================

Q1 = city_data['TAVG'].quantile(0.25)

Q3 = city_data['TAVG'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

city_data = city_data[
    (city_data['TAVG'] >= lower_bound)
    &
    (city_data['TAVG'] <= upper_bound)
]

print("\nAfter Outlier Removal:")
print(city_data.shape)

# ======================================================
# RESET INDEX
# ======================================================

city_data = city_data.reset_index(
    drop=True
)

# ======================================================
# SAVE CLEAN DATASET
# ======================================================

OUTPUT_FILE = "clean_chennai_temperature.csv"

city_data.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nClean dataset saved successfully!")

print("\nSaved File:", OUTPUT_FILE)

# ======================================================
# BASIC VISUALIZATION
# ======================================================

plt.figure(figsize=(14,6))

plt.plot(
    city_data['DATE'],
    city_data['TAVG']
)

plt.title(
    "Cleaned Chennai Temperature Time Series"
)

plt.xlabel("Year")

plt.ylabel("Temperature (°C)")

plt.grid(True)

plt.show()

# ======================================================
# FINAL PREVIEW
# ======================================================

print("\nFinal Dataset Preview:\n")

print(city_data.head())

print("\nFinal Shape:")

print(city_data.shape)