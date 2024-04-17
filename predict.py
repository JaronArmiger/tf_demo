# this version of predict.py
# concatenates the images into a single tensor

import os
import cv2
import glob
import time
import requests
import subprocess
import numpy as np
import tensorflow as tf
import concurrent.futures
from cog import BasePredictor, BaseModel, Input, Path
from typing import List
from tensorflow.python.client import device_lib


IMAGE_WIDTH_LANDSCAPE = 1080
IMAGE_HEIGHT_LANDSCAPE = 608
IMAGE_SHAPE_LANDSCAPE = (IMAGE_WIDTH_LANDSCAPE, IMAGE_HEIGHT_LANDSCAPE)

IMAGE_WIDTH_PORTRAIT = 608
IMAGE_HEIGHT_PORTRAIT = 1080
IMAGE_SHAPE_PORTRAIT = (IMAGE_WIDTH_PORTRAIT, IMAGE_HEIGHT_PORTRAIT)


class_names_landscape = [
    "34backleft",
    "34backright",
    "34frontleft",
    "34frontright",
    "autometer",
    "backseat",
    "carboot",
    "document",
    "engine",
    "front",
    "keys",
    "other",
    "rear",
    "rim",
    "seat_front",
    "side_left",
    "side_right",
    "tablet",
    "tire_tread",
    "wheel",
]

class_names_portrait = [
    "02_front_middle",
    "04_front_left_middle",
    "06_side_left_front_middle",
    "08_side_left_rear_middle",
    "10_rear_left_middle",
    "12_rear_middle",
    "14_rear_right_middle",
    "16_side_right_rear_middle",
    "18_side_right_front_middle",
    "20_front_right_middle",
    "47_no_car",
    "48_other",
    "49_black_screen",
]


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    devices = [x.name for x in local_device_protos]
    print(devices)


def set_env_vars():
    print("set_env_vars():")
    # Set CUDA_HOME environment variable
    cuda_path = "/usr/local/cuda/bin"
    cudnn_path = "/usr/local/cuda/lib64"
    cupti_path = "/usr/local/cuda/extras/CUPTI/lib64"
    cuda_11_8_path = "/usr/local/cuda-11.8/bin"
    cudnn_11_8_path = "/usr/local/cuda-11.8/lib64"
    cupti_11_8_path = "/usr/local/cuda-11.8/extras/CUPTI/lib64"
    cuda_home = "/usr/local/cuda-11.8"
    os.environ["CUDA_HOME"] = cuda_home

    # Update PATH environment variable
    os.environ["PATH"] = (
        f"{cuda_path}:{cudnn_path}:{cupti_path}:{cuda_11_8_path}:{cudnn_11_8_path}:{cupti_11_8_path}:{cuda_home}/bin:{os.getenv('PATH')}"
    )

    # Update LD_LIBRARY_PATH environment variable
    os.environ["LD_LIBRARY_PATH"] = (
        f"{cuda_path}:{cudnn_path}:{cupti_path}:{cuda_11_8_path}:{cudnn_11_8_path}:{cupti_11_8_path}:{cuda_home}/lib64:{os.getenv('LD_LIBRARY_PATH')}"
    )

    # Print environment variables
    print("CUDA_HOME:", os.getenv("CUDA_HOME"))
    print("CUDNN_HOME:", os.getenv("CUDNN_HOME"))
    print("PATH:", os.getenv("PATH"))
    print("LD_LIBRARY_PATH:", os.getenv("LD_LIBRARY_PATH"))


def cuda_checks():
    print("cuda_checks():")
    print("tf.test.is_built_with_cuda()")
    print(tf.test.is_built_with_cuda())
    # Call ls /usr/local/cuda*
    current_directory = os.getcwd()

    # Print the current working directory
    print("Current directory:", current_directory)
    path_value = os.getenv("PATH")
    print("PATH:", path_value)
    ld_library_path_value = os.getenv("LD_LIBRARY_PATH")
    print("LD_LIBRARY_PATH:", ld_library_path_value)
    cuda_directories = glob.glob("/usr/local/cuda*")

    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        )
        print("CUDA Version Information:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        )
        print("NVIDIA System Management Interface (nvidia-smi) Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
    cuda_path = os.getenv("CUDA_PATH")

    if cuda_path:
        print("CUDA Toolkit found at:", cuda_path)
        cuda_version = os.getenv("CUDA_VERSION")
        if cuda_version:
            cudnn_path = os.path.join(cuda_path, "extras", "CUPTI", "lib64")
            if os.path.exists(cudnn_path):
                print("cuDNN found at:", cudnn_path)
            else:
                print("cuDNN not found.")
        else:
            print("CUDA Toolkit is not installed, so cuDNN cannot be verified.")
    else:
        print("CUDA Toolkit not found.")

    # Check for cuDNN

    gpu_devices = tf.config.list_physical_devices("GPU")
    print("gpu_devices:")
    print(gpu_devices)
    if not gpu_devices:
        print("No GPU available. TensorFlow will use CPU.")
    else:
        print("GPU available. TensorFlow will use GPU.")


def preprocess_image(image_path, target_shape):
    img = download_image(image_path)
    img = cv2.resize(img, target_shape)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_batch = np.expand_dims(img_rgb, axis=0)
    return img_batch


def download_image(url):
    response = requests.get(url)
    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def download_multiple(images_list, IMAGE_SHAPE):
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = {
            executor.submit(preprocess_image, image_url, IMAGE_SHAPE): image_url
            for image_url in images_list
        }

        img_tensors = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
        return img_tensors


class Result(BaseModel):
    image_path: str
    position: str


class Predictor(BasePredictor):
    def setup(self) -> None:
        try:
            with tf.device("/gpu:0"):
                """Load the model into memory to make running multiple predictions efficient"""
                landscape_model_path = "models/position_analyser_landscape__tf_04"
                # self.landscape_model = tf.keras.models.load_model(landscape_model_path)
                self.landscape_model = tf.saved_model.load(landscape_model_path)
                portrait_model_path = "models/position_analyser_portrait__tf_06"
                # self.portrait_model = tf.keras.models.load_model(portrait_model_path)
                self.portrait_model = tf.saved_model.load(portrait_model_path)
        except Exception as e:
            print(e)

    def predict(
        self,
        # images: str = Input(description="images"),
        image: Path = Input(description="image", default=None),
        images: str = Input(description="images", default=None),
        mode: str = Input(description="landscape or portrait", default="portrait"),
        # scale: float = Input(
        #     description="Factor to scale image by", ge=0, le=10, default=1.5
        # ),
    ) -> List[Result]:
        try:
            print("TensorFlow version:", tf.__version__)
            set_env_vars()
            cuda_checks()
            get_available_devices()
            with tf.device("/gpu:0"):
                results_list: List[Result] = []
                if image is None and images is None:
                    return results_list
                elif image is None and images is not None:
                    images_list = images.split(", ")
                elif image is not None and images is None:
                    if mode == "portrait":
                        predictions = self.portrait_model.predict(image)
                    else:
                        predictions = self.landscape_model.predict(image)
                    predicted_class = np.argmax(predictions[i])
                    # results[image_path] = predicted_class
                    score = tf.nn.softmax(predictions[i])
                    selected_class = class_names[np.argmax(score)]
                    results_list.append(
                        Result(image_path=image_path, position=selected_class)
                    )
                    return results_list
                else:
                    images_list = images.split(", ")

                    if image is None and images is None:
                        return results_list
                    elif image is None and images is not None:
                        images_list = images.split(", ")
                    elif image is not None and images is None:
                        images_list = [image]
                    else:
                        images_list = images.split(", ")

                    if mode == "portrait":
                        IMAGE_SHAPE = IMAGE_SHAPE_PORTRAIT
                        class_names = class_names_portrait
                    else:
                        IMAGE_SHAPE = IMAGE_SHAPE_LANDSCAPE
                        class_names = class_names_landscape

                    start_download = time.time()
                    # img_tensors = [
                    #     preprocess_image(image_path, IMAGE_SHAPE)
                    #     for image_path in images_list
                    # ]
                    img_tensors = download_multiple(images_list, IMAGE_SHAPE)
                    # print("img_tensors:")
                    # print(img_tensors)
                    images_tensor = np.concatenate(img_tensors, axis=0)
                    end_download = time.time()
                    print(
                        f"Downloading images took: {end_download - start_download} seconds"
                    )

                    # Run inference on the concatenated tensor
                    start_prediction = time.time()
                    if mode == "portrait":
                        predictions = self.portrait_model.predict(images_tensor)
                    else:
                        predictions = self.landscape_model.predict(images_tensor)
                    end_prediction = time.time()
                    print(
                        f"Predictions took: {end_prediction - start_prediction} seconds"
                    )

                    for i, image_path in enumerate(images_list):
                        score = tf.nn.softmax(predictions[i])
                        selected_class = class_names[np.argmax(score)]
                        results_list.append(
                            Result(image_path=image_path, position=selected_class)
                        )

                    return results_list
        except Exception as e:
            print("exception occurred")
            print(e)
            return results_list
