from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

OPTICAL_FLOWS = {
    "dis": cv2.DISOpticalFlow_create(preset=cv2.DISOPTICAL_FLOW_PRESET_FAST),
    "farneback": cv2.optflow.createOptFlow_Farneback(),
    "deep_flow": cv2.optflow.createOptFlow_DeepFlow(),
    "pca_flow": cv2.optflow.createOptFlow_PCAFlow(),
    "dual_tvl1": cv2.optflow.createOptFlow_DualTVL1(),
    "dense_rlof": cv2.optflow.createOptFlow_DenseRLOF(),  # requires RGB input
}


@click.command()
@click.option(
    "--optical_flow",
    help="Add optical flow to input data",
    type=click.Choice(["dis", "farneback", "deep_flow", "pca_flow", "dual_tvl1", "dense_rlof"]),
    default=None,
)
@click.option(
    "--periods_path",
    help="Data frame with evaluation periods",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    default="data/Prepared/periods.pickle",
)
@click.option(
    "--dataset_path",
    help="Path to dataset image directory",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
    default="data/Folsom/images",
)
def generate_optical_flow(optical_flow, periods_path: Path, dataset_path: Path):
    Path(periods_path.parents[1], "flows", optical_flow).mkdir(parents=True, exist_ok=True)
    of = OPTICAL_FLOWS.get(optical_flow)

    with periods_path.open("rb") as f:
        periods = pd.read_pickle(f)

    for p in tqdm(periods):
        flow = None
        prev_image = None

        flow_x_path = Path(
            periods_path.resolve().parents[1], "flows", optical_flow, p["history"][-1]["image_name"].replace(".jpg", "_x.tiff")
        )
        flow_y_path = Path(
            periods_path.resolve().parents[1], "flows", optical_flow, p["history"][-1]["image_name"].replace(".jpg", "_y.tiff")
        )

        if flow_x_path.is_file() and flow_y_path.is_file():
            continue

        for history_item in p["history"]:
            image_path = Path(dataset_path, history_item["image_name"])
            source_image = np.asarray(Image.open(image_path))

            if optical_flow == "dense_rlof":
                if prev_image is None:
                    prev_image = source_image.copy()
                flow = of.calc(prev_image, source_image, flow)
                prev_image = source_image
            else:
                image_gray = cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY)
                if prev_image is None:
                    prev_image = image_gray.copy()
                flow = of.calc(prev_image, image_gray, flow)
                prev_image = image_gray

        flow_x = Image.fromarray(flow[..., 0])
        flow_y = Image.fromarray(flow[..., 1])
        flow_x.save(flow_x_path)
        flow_y.save(flow_y_path)


if __name__ == "__main__":
    generate_optical_flow()
