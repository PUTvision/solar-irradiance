import time
import logging
from pathlib import Path

import cv2
import click
import numpy as np
import onnxruntime as ort
import pandas as pd
import pytz
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
    "dense_rlof": cv2.optflow.createOptFlow_DenseRLOF(),    # requires RGB input
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

    return img_data.astype(np.float32)


@click.command()
@click.option("--model_path", help="ONNX model path", type=click.Path(exists=True, file_okay=True))
@click.option("--provider", help="Inference provider", type=click.Choice(["cpu", "openvino", "cuda", "tensorrt"]), default="cpu")
@click.option("--dims", help="Model dimensions: 3 for 3D, 2 for 2D models", type=int, default=2)
@click.option("--add_sun_mask", help="Add sun mask to input data", is_flag=True)
@click.option("--add_irradiance_channel", help="Add irradiance channel to input data", is_flag=True)
@click.option("--cloud_mask_method", help="Cloud segmentation method", type=click.Choice(["red_blue_ratio", "red_blue_difference", "normalized_blue_red_ratio"]), default=None)
@click.option("--optical_flow", help="Add optical flow to input data", type=click.Choice(["dis", "farneback", "deep_flow", "pca_flow", "dual_tvl1", "dense_rlof"]), default=None)
@click.option("--eval_periods_path", help="Data frame with evaluation periods", type=click.Path(exists=True, file_okay=True), default="data/Eval/eval_periods.pickle")
@click.option("--dataset_path", help="Path to dataset image directory", type=click.Path(exists=True, dir_okay=True), default="data/Eval/images")
def evaluate(model_path, dims, provider, add_sun_mask, add_irradiance_channel, cloud_mask_method, optical_flow, eval_periods_path, dataset_path):
    us_pacific = pytz.timezone('US/Pacific')
    utc = pytz.utc

    with open(eval_periods_path, "rb") as f:
        eval_periods = pd.read_pickle(f)

    inference_provider = PROVIDERS[provider]

    model_name = model_path.split("/")[-1]
    log.info(f"Inference model: {model_name}")
    log.info(f"ONNXRuntime provider: {inference_provider}")
    sess = ort.InferenceSession(model_path, providers=[inference_provider])

    inputs = [i.name for i in sess.get_inputs()]
    shapes = [s.shape for s in sess.get_inputs()]
    log.info(f"Model inputs: {inputs}")
    log.info(f"Model inputs' shapes: {shapes}")
    input_shape = [shapes[0][2], shapes[0][3]]

    sun_mask_gen = SunMask(LATITUDE, LONGITUDE, CAMERA_ORIENTATION_COMPENSATION, FOCAL_LENGTH)

    if cloud_mask_method is not None:
        cloud_mask = CloudMask(shape=input_shape, method=cloud_mask_method)

    of = OPTICAL_FLOWS.get(optical_flow)

    crop_mask = cv2.circle(
        np.zeros((input_shape[1], input_shape[0], 3), dtype=np.uint8),
        (input_shape[1] // 2, input_shape[0] // 2),
        input_shape[0] // 2,
        color=(1, 1, 1),
        thickness=-1,
    )

    # NN model warmup
    for _ in range(10):
        _ = sess.run(None, {inputs[idx]: np.random.normal(size=shapes[idx]).astype(np.float32) for idx in range(len(inputs))})

    target_irradiances = []
    outputs = []

    inference_time = []
    preprocessing_time = []
    process_time = []
    sun_mask_time = []
    irr_channel_time = []
    cloud_mask_time = []
    optical_flow_time = []

    for p in tqdm(eval_periods):
        process_start = time.time()

        source_images = []
        source_irradiances = []
        flow = None
        prev_image = None

        for history_item in p["history"]:
            image_path = Path(dataset_path, history_item["image_name"])
            irradiance = history_item["irradiance"] #/ MAX_IRRADIANCE
            source_image = np.asarray(Image.open(image_path))

            preprocessing_start = time.time()
            source_image = cv2.resize(source_image, input_shape)
            source_image = np.where(crop_mask.astype(bool), source_image, 0)
            input_data = preprocess(source_image)
            preprocessing_time.append(time.time() - preprocessing_start)

            if add_sun_mask:
                date = pd.to_datetime(image_path.name[:15], format='%Y%m%d_%H%M%S')
                us_pacific_date = us_pacific.localize(date)
                utc_date = us_pacific_date.astimezone(utc).strftime('%Y%m%d_%H%M%S')

                sun_mask_start = time.time()
                sun_mask = sun_mask_gen(image_shape=input_shape, timestamp=utc_date)
                input_data = np.concatenate([input_data, np.transpose(sun_mask, (2, 0, 1))], axis=0)
                sun_mask_time.append(time.time() - sun_mask_start)

            if add_irradiance_channel:
                irr_channel_start = time.time()
                if add_sun_mask:
                    input_data[-1] *= irradiance
                else:
                    input_data = np.concatenate(
                        [input_data, np.ones((1, *input_shape), dtype=np.float32) * irradiance],
                        axis=0,
                    )
                irr_channel_time.append(time.time() - irr_channel_start)

            if cloud_mask_method is not None:
                cloud_mask_start = time.time()
                mask = cloud_mask(image=source_image)
                input_data = np.concatenate([input_data, np.transpose(mask, (2, 0, 1))], axis=0)
                cloud_mask_time.append(time.time() - cloud_mask_start)

            if optical_flow is not None:
                of_time_start = time.time()
                image = cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY) if optical_flow != "dense_rlof" else source_image
                if prev_image is None:
                    prev_image = image.copy()
                flow = of.calc(prev_image, image, flow)
                prev_image = image
                input_data = np.concatenate([input_data, np.transpose(flow, (2, 0, 1))], axis=0)
                optical_flow_time.append(time.time() - of_time_start)

            source_images.append(input_data)
            source_irradiances.append(irradiance)

        target_irradiance = p["target_irradiance"] #/ MAX_IRRADIANCE

        image_input = np.expand_dims(np.transpose(source_images, (1, 0, 2, 3)), axis=0).astype(np.float32) if dims == 3 else np.array(source_images[-1:], dtype=np.float32)

        inference_start = time.time()
        output = sess.run(None, {
            inputs[0]: image_input,
            inputs[1]: np.expand_dims(source_irradiances, axis=0).astype(np.float32),
        })[0][0][0]
        loop_end = time.time()
        inference_time.append(loop_end - inference_start)
        process_time.append(loop_end - process_start)

        target_irradiances.append(target_irradiance)
        outputs.append(output)

    periods_num = len(eval_periods)
    log.info(f"Inference average time [s]: {np.mean(inference_time)}")
    log.info(f"Inference std time [s]: {np.std(inference_time)}")
    log.info("-"*50)
    log.info(f"Preprocessing average time [s]: {np.mean(preprocessing_time)}")
    log.info(f"Preprocessing std time [s]: {np.std(preprocessing_time)}")
    log.info("-"*50)
    log.info(f"Process average time [s]: {np.mean(process_time)}")
    log.info(f"Process std time [s]: {np.std(process_time)}")
    log.info("-"*50)
    log.info(f"Sun mask generation average time [s]: {np.mean(sun_mask_time)}")
    log.info(f"Sun mask generation std time [s]: {np.std(sun_mask_time)}")
    log.info("-"*50)
    log.info(f"Irradiance channel addition average time [s]: {np.mean(irr_channel_time)}")
    log.info(f"Irradiance channel addition std time [s]: {np.std(irr_channel_time)}")
    log.info("-"*50)
    log.info(f"Cloud mask average time [s]: {np.mean(cloud_mask_time)}")
    log.info(f"Cloud mask std time [s]: {np.std(cloud_mask_time)}")
    log.info("-"*50)
    log.info(f"Optical flow average time [s]: {np.mean(optical_flow_time)}")
    log.info(f"Optical flow std time [s]: {np.std(optical_flow_time)}")

    mape = mean_absolute_percentage_error(target_irradiances, outputs)
    log.info(f"MAPE [%]: {mape*100:.2f}")


if __name__ == "__main__":
    evaluate()
