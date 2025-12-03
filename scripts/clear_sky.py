import click
import pandas as pd
import pvlib

# Folsom, CA Coordinates and dataset time range
LATITUDE_FOLSOM = 38.642
LONGITUDE_FOLSOM = -121.148
TIMEZONE_FOLSOM = "US/Pacific"
START_FOLSOM = "2013-12-31 16:00:00"
END_FOLSOM = "2016-12-31 16:00:00"


@click.command()
@click.option("--latitude", default=LATITUDE_FOLSOM, help="Latitude of the location.")
@click.option("--longitude", default=LONGITUDE_FOLSOM, help="Longitude of the location.")
@click.option("--timezone", default=TIMEZONE_FOLSOM, help="Timezone of the location.")
@click.option("--start", default=START_FOLSOM, help="Start time (local). Format `YYYY-MM-DD HH:MM:SS`")
@click.option("--end", default=END_FOLSOM, help="End time (local). Format `YYYY-MM-DD HH:MM:SS`")
def generate_clear_sky_reference(latitude, longitude, timezone, start, end):
    print(f"Generating clear-sky data for ({latitude}, {longitude}) using Ineichen-Perez model.")
    print(f"Time range (local): {start} to {end}")

    # Create the 1-minute time index for the specified day
    start = pd.Timestamp(start, tz=timezone)
    end = pd.Timestamp(end, tz=timezone)
    times = pd.date_range(start, end, freq="1min", tz=timezone)

    # Create a pvlib.Location object
    location = pvlib.location.Location(latitude, longitude, tz=timezone)

    # Get solar position (zenith, azimuth, etc.) for the times
    solar_pos = location.get_solarposition(times)

    # Get atmospheric data (Linke Turbidity) from pvlib's built-in dataset
    linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)

    # Calculate clear-sky GHI using the Ineichen-Perez model
    clearsky = location.get_clearsky(times, model="ineichen", solar_position=solar_pos, linke_turbidity=linke_turbidity)

    # Rename 'ghi' to 'ghi_clear' for clarity.
    clearsky_data = clearsky.rename(columns={"ghi": "ghi_clear"})
    clearsky_data.drop(columns=["dni", "dhi"], inplace=True)

    # Ensure nighttime GHI is 0 and not NaN
    clearsky_data["ghi_clear"] = clearsky_data["ghi_clear"].fillna(0).clip(lower=0)

    # Remove timezone info for saving
    clearsky_data.index = clearsky_data.index.tz_localize(None)

    print("\nSuccessfully generated local clear-sky data using Ineichen model.")
    print("\n--- Data Head ---")
    print(clearsky_data.head(3))

    output_path = "data/prepared/clear_sky_ineichen_poznan.csv"
    clearsky_data.to_csv(output_path, index_label="datetime")
    print(f"\nClear-sky data saved to '{output_path}'")


if __name__ == "__main__":
    generate_clear_sky_reference()
