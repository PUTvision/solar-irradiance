from pathlib import Path
from typing import Any
from multiprocessing import Pool, cpu_count
from functools import partial

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pvlib
import pytz
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class HendrikxFeatureComputer:
    latitude = 38.642
    longitude = -121.148
    altitude = 68  # Altitude in meters
    us_pacific = pytz.timezone("US/Pacific")
    utc = pytz.utc

    def __init__(self, data_root: Path):
        self._data_root = data_root

    @staticmethod
    def get_brightness(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_image)
        return brightness

    @staticmethod
    def get_cloud_pixels(image, rbr_threshold=0.8):
        b_channel, g_channel, r_channel = cv2.split(image)
        r_float = r_channel.astype(float)
        b_float = b_channel.astype(float)
        rbr = r_float / (b_float + 1e-6)
        cloud_mask = rbr > rbr_threshold
        cloud_pixel_count = np.sum(cloud_mask)
        return int(cloud_pixel_count)

    @staticmethod
    def get_edge_count(image, canny_threshold1=100, canny_threshold2=200):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, canny_threshold1, canny_threshold2)
        edge_count = np.count_nonzero(edges)
        return int(edge_count)

    @staticmethod
    def get_corner_count(image, harris_threshold_ratio=0.01):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_float = np.float32(gray_image)
        corner_scores = cv2.cornerHarris(gray_float, 2, 3, 0.04)
        threshold = harris_threshold_ratio * corner_scores.max()
        corner_mask = corner_scores > threshold
        corner_count = np.sum(corner_mask)
        return int(corner_count)

    @staticmethod
    def get_clear_sky_values(timestamp, latitude, longitude, altitude):
        location = pvlib.location.Location(
            latitude=latitude, longitude=longitude, altitude=altitude
        )
        if isinstance(timestamp, pd.Timestamp):
            timestamp = pd.DatetimeIndex([timestamp])
        clearsky = location.get_clearsky(timestamp)
        return clearsky["ghi"].iloc[0]

    @staticmethod
    def calculate_csi(measured_ghi, clear_sky_ghi):
        if clear_sky_ghi == 0:
            return 0.0
        csi = measured_ghi / clear_sky_ghi
        csi = min(csi, 1.2)
        if np.isinf(csi):
            return 0.0
        return csi

    @staticmethod
    def get_solar_features(timestamp, latitude, longitude, altitude):
        if isinstance(timestamp, pd.Timestamp):
            timestamp = pd.DatetimeIndex([timestamp])
        solar_pos = pvlib.solarposition.get_solarposition(
            time=timestamp,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
        )
        return (
            solar_pos["zenith"].iloc[0],
            solar_pos["azimuth"].iloc[0],
            solar_pos["apparent_elevation"].iloc[0],
        )

    def compute_all_features_to_dataframe(
        self, 
        periods: list[dict[str, Any]], 
        output_path: Path = None,
        n_processes: int = None
    ) -> pd.DataFrame:
        """
        Compute features for all unique images in periods using multiprocessing.
        
        Args:
            periods: List of period dictionaries with 'history' and 'target_irradiance'
            output_path: Optional path to save the DataFrame (as CSV or parquet)
            n_processes: Number of processes to use (defaults to cpu_count() - 1)
            
        Returns:
            pd.DataFrame with image_name as index and computed features as columns
        """
        # Collect all unique images with their irradiance values
        image_data = {}
        
        for period in tqdm(periods, desc="Collecting images"):
            for history_item in period["history"]:
                image_name = history_item["image_name"]
                if image_name not in image_data:
                    image_data[image_name] = history_item["irradiance"]

        # Prepare data for multiprocessing
        image_items = list(image_data.items())
        
        # Set number of processes
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)
        
        print(f"Processing {len(image_items)} unique images using {n_processes} processes...")
        
        # Create partial function with fixed arguments
        compute_func = partial(
            _compute_features_worker,
            data_root=self._data_root,
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude,
            us_pacific=self.us_pacific
        )
        
        # Use multiprocessing with progress bar
        with Pool(processes=n_processes) as pool:
            all_features = list(tqdm(
                pool.imap(compute_func, image_items),
                total=len(image_items),
                desc="Computing features"
            ))
        
        # Filter out failed computations
        all_features = [f for f in all_features if f is not None]

        # Create DataFrame with image_name as index
        df = pd.DataFrame(all_features)

        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df.drop(columns=["image_name"])), columns=df.columns[1:])
        df_normalized.insert(0, "image_name", df["image_name"])
        df_normalized = df_normalized.set_index("image_name")

        if output_path:
            output_path = Path(output_path)
            df.to_csv(output_path)
            print(f"Features saved to {output_path}")

        return df_normalized


def _compute_features_worker(image_item, data_root, latitude, longitude, altitude, us_pacific):
    """
    Worker function for multiprocessing. Must be at module level for pickling.
    
    Args:
        image_item: Tuple of (image_name, irradiance)
        data_root: Path to data root directory
        latitude: Location latitude
        longitude: Location longitude
        altitude: Location altitude
        us_pacific: Timezone object
        
    Returns:
        Dictionary with computed features or None if error
    """
    image_name, irradiance = image_item
    
    try:
        image_path = data_root / "images" / image_name
        image = np.asarray(Image.open(image_path))

        brightness = HendrikxFeatureComputer.get_brightness(image)
        cloud_pixels = HendrikxFeatureComputer.get_cloud_pixels(image)
        edge_count = HendrikxFeatureComputer.get_edge_count(image)
        corner_count = HendrikxFeatureComputer.get_corner_count(image)

        timestamp = pd.to_datetime(image_path.stem, format="%Y%m%d_%H%M%S")
        timestamp = timestamp.tz_localize(us_pacific)
        
        clear_sky_values = HendrikxFeatureComputer.get_clear_sky_values(
            timestamp, latitude, longitude, altitude
        )
        csi = HendrikxFeatureComputer.calculate_csi(irradiance, clear_sky_values)
        zenith, azimuth, sun_earth_distance = HendrikxFeatureComputer.get_solar_features(
            timestamp, latitude, longitude, altitude
        )

        return {
            "image_name": image_name,
            "irradiance": irradiance,
            "brightness": brightness,
            "cloud_pixels": cloud_pixels,
            "edge_count": edge_count,
            "corner_count": corner_count,
            "clear_sky_ghi": clear_sky_values,
            "csi": csi,
            "zenith": zenith,
            "azimuth": azimuth,
            "apparent_elevation": sun_earth_distance,
        }
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return None


if __name__ == "__main__":
    import pickle
    
    # Example usage
    data_root = Path("/home/mateusz.piechocki/solar-irradiance/data/prepared")
    periods_path = Path("/home/mateusz.piechocki/solar-irradiance/data/prepared/periods.pickle")
    
    # Load periods
    with periods_path.open("rb") as f:
        periods = pd.read_pickle(f)
    
    print(f"Loaded {len(periods)} periods")
    
    # Compute features using all available cores minus 1
    computer = HendrikxFeatureComputer(data_root)
    features_df = computer.compute_all_features_to_dataframe(
        periods, 
        output_path=data_root / "hendrikx_features.csv",
        n_processes=None  # Uses cpu_count() - 1
    )
    
    print(f"\nDataFrame shape: {features_df.shape}")
    print(f"\nColumns: {features_df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(features_df.head())
    print(f"\nDataFrame info:")
    print(features_df.info())
