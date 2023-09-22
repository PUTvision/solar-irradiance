import time
import logging
from pathlib import Path

import cv2
import pandas as pd
import click
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm

from solar_irradiance.datamodules.sun_mask import SunMask
from solar_irradiance.datamodules.cloud_mask import CloudMask


fromat = "[%(levelname)s] [%(module)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=fromat)
log = logging.getLogger(__name__)


LATITUDE = 38.642
LONGITUDE = -121.148
CAMERA_ORIENTATION_COMPENSATION = 165
FOCAL_LENGTH = 0.48

MAX_IRRADIANCE = 1466.0     # max irradiance in the dataset

OPTICAL_FLOWS = {
    "dis": cv2.DISOpticalFlow_create(preset=cv2.DISOPTICAL_FLOW_PRESET_FAST),
    "farneback": cv2.optflow.createOptFlow_Farneback(),
    "deep_flow": cv2.optflow.createOptFlow_DeepFlow(),
    "pca_flow": cv2.optflow.createOptFlow_PCAFlow(),
    "dual_tvl1": cv2.optflow.createOptFlow_DualTVL1(),
    "dense_rlof": cv2.optflow.createOptFlow_DenseRLOF(), # requires RGB input
}

PROVIDERS = {
    "cpu": ("CPUExecutionProvider"),
    "openvino": ("OpenVINOExecutionProvider", {"device_type": "MYRIAD_FP16"}),
    "cuda": ("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'}),
    "tensorrt": (
        "TensorrtExecutionProvider",
        {
            "device_id": 0,
            "trt_fp16_enable": False,
            "trt_int8_enable": False,
            "trt_int8_use_native_calibration_table": True,
            "trt_engine_cache_enable": False,
        },
    ),
}


def preprocess(img_data: np.ndarray) -> np.ndarray:
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])

    img_data = (img_data / 255 - mean_vec) / stddev_vec
    img_data = np.transpose(img_data, (2, 0, 1))

    return img_data.astype("float32")


@click.command()
@click.option("--model_path", help="ONNX model path", type=click.Path(exists=True, file_okay=True))
@click.option("--provider", help="Inference provider", type=click.Choice(["cpu", "openvino", "cuda", "tensorrt"]), default="cpu")
@click.option("--dims", help="Model dimensions: 3 for 3D, 2 for 2D models", type=int, default=2)
@click.option("--add_sun_mask", help="Add sun mask to input data", is_flag=True)
@click.option("--add_irradiance_channel", help="Add irradiance channel to input data", is_flag=True)
@click.option("--cloud_mask_method", help="Cloud segmentation method", type=click.Choice(["blue_red_ratio", "blue_red_difference", "normalized_blue_red_ratio"]), default=None)
@click.option("--optical_flow", help="Add optical flow to input data", type=click.Choice(["dis", "farneback", "deep_flow", "pca_flow", "dual_tvl1", "dense_rlof"]), default=None)
@click.option("--eval_periods_path", help="Data frame with evaluation periods", type=click.Path(exists=True, file_okay=True), default="data/Eval/eval_periods.pickle")
@click.option("--dataset_path", help="Path to dataset image directory", type=click.Path(exists=True, dir_okay=True), default="data/Eval/images")
def main(model_path, dims, provider, add_sun_mask, add_irradiance_channel, cloud_mask_method, optical_flow, eval_periods_path, dataset_path):
    of = OPTICAL_FLOWS.get(optical_flow)
    input_shape = (384, 384)

    with open(eval_periods_path, "rb") as f:
        eval_periods = pd.read_pickle(f)

    sun_mask_gen = SunMask(LATITUDE, LONGITUDE, CAMERA_ORIENTATION_COMPENSATION, FOCAL_LENGTH)

    if cloud_mask_method is not None:
        cloud_mask = CloudMask(shape=input_shape, method=cloud_mask_method)

    inference_provider = PROVIDERS[provider]

    model_name = model_path.split("/")[-1]
    log.info(f"Inference model: {model_name}")
    log.info(f"ONNXRuntime provider: {inference_provider}")
    sess = ort.InferenceSession(model_path, providers=[inference_provider])

    inputs = [l.name for l in sess.get_inputs()]
    shapes = [l.shape for l in sess.get_inputs()]
    log.info(f"Model inputs: {inputs}")
    log.info(f"Model inputs' shapes: {shapes}")

    # NN model warmup
    for _ in range(10):
        _ = sess.run(None, {inputs[idx]: np.random.normal(size=shapes[idx]).astype(np.float32) for idx in range(len(inputs))})[0]

    target_irradiances = []
    outputs = []

    inference_time = 0.0
    process_time = 0.0
    sun_mask_time = 0.0
    irr_channel_time = 0.0
    cloud_mask_time = 0.0
    optical_flow_time = 0.0

    for p in tqdm(eval_periods):
        process_start = time.time()

        source_images = []
        source_irradiances = []
        flow = None
        prev_image = None

        for history_item in p["history"]:
            image_path = Path(dataset_path, history_item["image_name"])
            irradiance = history_item["irradiance"] / MAX_IRRADIANCE
            source_image = np.asarray(Image.open(image_path))
            source_image = cv2.resize(source_image, input_shape)

            input_data = preprocess(source_image)

            if add_sun_mask:
                sun_mask_start = time.time()
                sun_mask = sun_mask_gen(image=source_image, timestamp=image_path.name[:15])
                input_data = np.concatenate([input_data, np.transpose(sun_mask, (2, 0, 1))], axis=0)
                sun_mask_time += time.time() - sun_mask_start

            if add_irradiance_channel:
                irr_channel_start = time.time()
                if add_sun_mask:
                    input_data[-1] *= irradiance
                else:
                    input_data = np.concatenate([input_data, np.ones((1, *input_shape), dtype=np.float32) * irradiance], axis=0)
                irr_channel_time += time.time() - irr_channel_start

            if cloud_mask_method is not None:
                cloud_mask_start = time.time()
                mask = cloud_mask(image=source_image)
                input_data = np.concatenate([input_data, np.transpose(mask, (2, 0, 1))], axis=0)
                cloud_mask_time += time.time() - cloud_mask_start

            if optical_flow is not None:
                of_time_start = time.time()
                image = cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY) if optical_flow != "dense_rlof" else source_image
                if prev_image is None:
                    prev_image = image.copy()
                flow = of.calc(prev_image, image, flow)
                prev_image = image
                input_data = np.concatenate([input_data, np.transpose(flow, (2, 0, 1))], axis=0)
                optical_flow_time += time.time() - of_time_start

            source_images.append(input_data)
            source_irradiances.append(irradiance)

        target_irradiance = p["target_irradiance"] / MAX_IRRADIANCE

        image_input = np.expand_dims(np.transpose(source_images, (1, 0, 2, 3)), axis=0).astype(np.float32) if dims == 3 else np.array(source_images[-1:], dtype=np.float32)

        inference_start = time.time()
        output = sess.run(None, {
            inputs[0]: image_input,
            inputs[1]: np.expand_dims(source_irradiances, axis=0).astype(np.float32),
        })[0][0][0]
        loop_end = time.time()
        inference_time += loop_end - inference_start
        process_time += loop_end - process_start

        target_irradiances.append(target_irradiance)
        outputs.append(output)

    periods_num = len(eval_periods)
    log.info(f"Inference average time [s]: {inference_time / periods_num}")
    log.info(f"Inference frames per second [img/s]: {periods_num / inference_time}")
    log.info(f"Process average time [s]: {process_time / periods_num}")
    log.info(f"Process frames per second [img/s]: {periods_num / process_time}")
    log.info(f"Sun mask generation average time [s]: {sun_mask_time / periods_num}")
    log.info(f"Irradiance channel addition average time [s]: {irr_channel_time / periods_num}")
    log.info(f"Cloud mask average time [s]: {cloud_mask_time / periods_num}")
    log.info(f"Optical flow average time [s]: {optical_flow_time / periods_num}")

    mape = mean_absolute_percentage_error(target_irradiances, outputs)
    log.info(f"MAPE [%]: {mape*100:.2f}")


if __name__ == "__main__":
    main()
