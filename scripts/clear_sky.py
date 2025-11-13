import pandas as pd
import pvlib

# --- 1. Configuration ---

# Folsom, CA Coordinates
LATITUDE = 38.642
LONGITUDE = -121.148
TIMEZONE = "US/Pacific"

# Define the time range you want to analyze (local time)
# Using a recent full day for this example
start_local = pd.Timestamp("2013-12-31 16:00:00", tz=TIMEZONE)
end_local = pd.Timestamp("2016-12-31 16:00:00", tz=TIMEZONE)

print(f"Generating clear-sky data for Folsom, CA ({LATITUDE}, {LONGITUDE})")
print("Model: Ineichen-Perez (via pvlib.location.get_clearsky)")
print(f"Time range (local): {start_local} to {end_local}")

# --- 2. Generate Time Series and Location Object ---

# Create a pvlib.Location object
location = pvlib.location.Location(LATITUDE, LONGITUDE, tz=TIMEZONE)

# Create the 1-minute time index for the specified day
times = pd.date_range(start_local, end_local, freq="1min", tz=TIMEZONE)

# --- 3. Get Local Clear-Sky Data (Ineichen Model) ---
try:
    # Get solar position (zenith, azimuth, etc.) for the times
    solar_pos = location.get_solarposition(times)

    # Get atmospheric data (Linke Turbidity) from pvlib's built-in dataset
    # This is a key input for the Ineichen model
    linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times, LATITUDE, LONGITUDE)

    # Calculate clear-sky GHI using the Ineichen-Perez model
    # pvlib's get_clearsky() function uses 'ineichen' model by default
    # It requires the solar position and Linke turbidity
    clearsky = location.get_clearsky(times, model="ineichen", solar_position=solar_pos, linke_turbidity=linke_turbidity)

    # The output is a DataFrame. Let's rename 'ghi' to 'ghi_clear' for clarity.
    clearsky_data = clearsky.rename(columns={"ghi": "ghi_clear"})
    clearsky_data.drop(columns=["dni", "dhi"], inplace=True)

    # Ensure nighttime GHI is 0 and not NaN
    clearsky_data["ghi_clear"] = clearsky_data["ghi_clear"].fillna(0).clip(lower=0)

    print("\nSuccessfully generated local clear-sky data using Ineichen model.")
    print("\n--- Data Head ---")
    print(clearsky_data.head(3))

    clearsky_data.to_csv("../data/prepared/clear_sky_ineichen.csv", index_label="datetime")
    print("\nClear-sky data saved to '../data/prepared/clear_sky_ineichen.csv'")

except Exception as e:
    print("\n--- ERROR ---")
    print(f"An error occurred during the local clear-sky calculation: {e}")
