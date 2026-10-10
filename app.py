
import os
import gc
import time
import uuid
import logging

import cv2
import numpy as np
import torch

from flask import Flask, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.transforms import functional as TF
from skimage.metrics import structural_similarity as ssim

from model import TransUNetV2


# ============================================================
# PERFORMANCE AND MEMORY SETTINGS
# ============================================================

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)

IMAGE_SIZE = 512
IS_RENDER = bool(os.environ.get("RENDER"))

# Smaller batches on Render prevent long-running requests.
MAX_BATCH_SIZE = 8 if IS_RENDER else 45

# Maximum combined HTTP upload size.
MAX_UPLOAD_MB = 40 if IS_RENDER else 200

# Reject excessively large individual images.
MAX_IMAGE_PIXELS = 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
}


# ============================================================
# FLASK SETUP
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = (
    MAX_UPLOAD_MB * 1024 * 1024
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR, "static", "uploads"
)

RESULT_FOLDER = os.path.join(
    BASE_DIR, "static", "results"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RetinaSeg")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

device = torch.device("cpu")
DEVICE_NAME = "CPU"

MODEL_PATH = os.path.join(
    BASE_DIR, "combined_transunet_v2.pt"
)

logger.info("Using device: %s", device)

model = TransUNetV2()

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict, strict=True)

del state_dict
gc.collect()

model.eval()

logger.info("Trained TransUNet model loaded successfully")


# ============================================================
# IMAGE UTILITIES
# ============================================================

def valid_extension(filename):
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def get_file_key(filename):
    name = os.path.splitext(
        os.path.basename(filename)
    )[0].lower().strip()

    suffixes = [
        "_manual1",
        "_manual",
        "_ground_truth",
        "_groundtruth",
        "_gt",
        "_mask"
    ]

    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    return name


def build_ground_truth_lookup(files):
    lookup = {}

    for file in files:
        if file and file.filename:
            lookup[get_file_key(file.filename)] = file

    return lookup


def load_image(file, mode="RGB"):
    if not valid_extension(file.filename):
        raise ValueError(
            f"Unsupported image format: {file.filename}"
        )

    file.stream.seek(0)

    with Image.open(file.stream) as source:
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Image is too large: {file.filename}"
            )

        image = ImageOps.exif_transpose(source)
        image = image.convert(mode)
        image.load()

    return image


def resize_image(image, nearest=False):
    method = (
        Image.Resampling.NEAREST
        if nearest
        else Image.Resampling.BILINEAR
    )

    return image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        method
    )


# ============================================================
# RETINAL IMAGE ENHANCEMENT
# ============================================================

def enhance_retina_rgb(img_pil):
    arr = np.asarray(
        img_pil.convert("RGB"),
        dtype=np.uint8
    )

    green = arr[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_green = clahe.apply(green)

    output = arr.astype(np.float32)

    output[:, :, 1] = (
        0.35 * output[:, :, 1]
        + 0.65 * enhanced_green
    )

    output[:, :, 0] = (
        0.80 * output[:, :, 0]
        + 0.20 * enhanced_green
    )

    output[:, :, 2] = (
        0.80 * output[:, :, 2]
        + 0.20 * enhanced_green
    )

    output = np.clip(
        output, 0, 255
    ).astype(np.uint8)

    return Image.fromarray(output)


def preprocess_image(image):
    enhanced = enhance_retina_rgb(image)

    tensor = TF.to_tensor(enhanced)

    tensor = TF.normalize(
        tensor,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

    enhanced.close()

    return tensor.unsqueeze(0)


# ============================================================
# EVALUATION METRICS
# ============================================================

def calculate_dice(pred, truth):
    pred_binary = pred > 0
    truth_binary = truth > 0

    intersection = np.logical_and(
        pred_binary,
        truth_binary
    ).sum()

    denominator = (
        pred_binary.sum()
        + truth_binary.sum()
    )

    if denominator == 0:
        return 1.0

    return float(
        2.0 * intersection / denominator
    )


def calculate_ssim(pred, truth):
    return float(
        ssim(
            pred.astype(np.uint8),
            truth.astype(np.uint8),
            data_range=255
        )
    )


# ============================================================
# MODEL INFERENCE
# ============================================================

def segment_retina(image, threshold):
    """
    All outputs remain at 512x512 to avoid large
    temporary arrays and excessive memory usage.
    """

    tensor = None
    logits = None
    probabilities = None

    try:
        tensor = preprocess_image(image)

        start = time.perf_counter()

        with torch.inference_mode():
            logits = model(tensor)

            probabilities = torch.sigmoid(logits)

            probability_map = (
                probabilities[0, 0]
                .detach()
                .cpu()
                .numpy()
                .copy()
            )

        inference_time = (
            time.perf_counter() - start
        ) * 1000.0

        # Release PyTorch tensors before image rendering.
        del tensor, logits, probabilities

        tensor = None
        logits = None
        probabilities = None

        mask = (
            probability_map >= threshold
        ).astype(np.uint8) * 255

        probability_uint8 = np.clip(
            probability_map * 255.0,
            0,
            255
        ).astype(np.uint8)

        # Create heatmap.
        heatmap_bgr = cv2.applyColorMap(
            probability_uint8,
            cv2.COLORMAP_TURBO
        )

        heatmap_rgb = cv2.cvtColor(
            heatmap_bgr,
            cv2.COLOR_BGR2RGB
        )

        # Create overlay without a second full-size
        # vessel layer.
        original_np = np.asarray(
            image,
            dtype=np.uint8
        )

        overlay = original_np.copy()

        vessel_pixels = mask > 0

        if np.any(vessel_pixels):
            selected = overlay[vessel_pixels]

            overlay[vessel_pixels] = (
                0.45 * selected
                + 0.55 * np.array(
                    [0, 255, 255],
                    dtype=np.float32
                )
            ).astype(np.uint8)

            del selected

        mask_image = Image.fromarray(mask)
        heatmap_image = Image.fromarray(heatmap_rgb)
        overlay_image = Image.fromarray(overlay)

        del probability_map
        del probability_uint8
        del heatmap_bgr
        del heatmap_rgb
        del original_np
        del overlay
        del vessel_pixels
        del mask

        return (
            mask_image,
            heatmap_image,
            overlay_image,
            inference_time
        )

    finally:
        # If inference fails, release temporary tensors.
        tensor = None
        logits = None
        probabilities = None


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_single_image(
    file,
    gt_file,
    threshold,
    number,
    total
):
    image = None
    mask_image = None
    heatmap_image = None
    overlay_image = None
    gt_image = None

    try:
        original_filename = file.filename

        logger.info(
            "[%d/%d] Processing: %s",
            number,
            total,
            original_filename
        )

        # Resize immediately to control memory.
        source = load_image(file, "RGB")

        try:
            image = resize_image(source)
        finally:
            source.close()

        run_id = uuid.uuid4().hex[:12]

        original_name = f"{run_id}_original.png"
        mask_name = f"{run_id}_mask.png"
        overlay_name = f"{run_id}_overlay.png"
        heatmap_name = f"{run_id}_heatmap.png"

        image.save(
            os.path.join(UPLOAD_FOLDER, original_name),
            optimize=True
        )

        (
            mask_image,
            heatmap_image,
            overlay_image,
            inference_time
        ) = segment_retina(image, threshold)

        mask_image.save(
            os.path.join(RESULT_FOLDER, mask_name),
            optimize=True
        )

        overlay_image.save(
            os.path.join(RESULT_FOLDER, overlay_name),
            optimize=True
        )

        heatmap_image.save(
            os.path.join(RESULT_FOLDER, heatmap_name),
            optimize=True
        )

        image_dice = None
        image_ssim = None
        matched_gt_name = None

        if gt_file is not None:
            matched_gt_name = gt_file.filename

            gt_source = load_image(gt_file, "L")

            try:
                gt_image = resize_image(
                    gt_source,
                    nearest=True
                )
            finally:
                gt_source.close()

            gt_np = (
                np.asarray(gt_image) > 127
            ).astype(np.uint8) * 255

            pred_np = np.asarray(mask_image)

            image_dice = round(
                calculate_dice(pred_np, gt_np),
                4
            )

            image_ssim = round(
                calculate_ssim(pred_np, gt_np),
                4
            )

            logger.info(
                "GT: %s | Dice: %.4f | SSIM: %.4f",
                matched_gt_name,
                image_dice,
                image_ssim
            )

            del gt_np
            del pred_np

        # Store only strings and numbers in results.
        return {
            "filename": original_filename,
            "ground_truth": matched_gt_name,
            "original_image": "uploads/" + original_name,
            "result_image": "results/" + mask_name,
            "overlay_image": "results/" + overlay_name,
            "heatmap_image": "results/" + heatmap_name,
            "inference_time": round(inference_time, 1),
            "dice": image_dice,
            "ssim_score": image_ssim
        }

    finally:
        for img in (
            image,
            mask_image,
            heatmap_image,
            overlay_image,
            gt_image
        ):
            if img is not None:
                img.close()

        gc.collect()


# ============================================================
# MAIN PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    data = {
        "original_image": None,
        "result_image": None,
        "overlay_image": None,
        "heatmap_image": None,
        "dice": None,
        "ssim_score": None,
        "inference_time": None,
        "threshold": 0.80,
        "dataset": "General",
        "device_name": DEVICE_NAME,
        "error": None,
        "batch_results": [],
        "batch_count": 0,
        "average_dice": None,
        "average_ssim": None
    }

    if request.method == "POST":

        try:
            files = [
                file
                for file in request.files.getlist("image")
                if file and file.filename
            ]

            if not files:
                raise ValueError(
                    "Please select at least one retinal image."
                )

            if len(files) > MAX_BATCH_SIZE:
                raise ValueError(
                    f"Please upload a maximum of "
                    f"{MAX_BATCH_SIZE} images per batch. "
                    f"This limit prevents server memory "
                    f"and timeout errors."
                )

            dataset = request.form.get(
                "dataset",
                "General"
            )

            data["dataset"] = dataset

            threshold = float(
                request.form.get(
                    "threshold",
                    0.80
                )
            )

            if not np.isfinite(threshold):
                raise ValueError(
                    "Threshold must be a valid number."
                )

            threshold = max(
                0.0,
                min(1.0, threshold)
            )

            data["threshold"] = threshold

            ground_truth_files = [
                file
                for file in request.files.getlist(
                    "ground_truth"
                )
                if file and file.filename
            ]

            gt_lookup = build_ground_truth_lookup(
                ground_truth_files
            )

            logger.info(
                "Starting batch: %d images, %d GT masks",
                len(files),
                len(ground_truth_files)
            )

            batch_results = []
            dice_scores = []
            ssim_scores = []

            for number, file in enumerate(
                files,
                start=1
            ):
                image_key = get_file_key(
                    file.filename
                )

                gt_file = gt_lookup.get(image_key)

                result = process_single_image(
                    file=file,
                    gt_file=gt_file,
                    threshold=threshold,
                    number=number,
                    total=len(files)
                )

                batch_results.append(result)

                if result["dice"] is not None:
                    dice_scores.append(
                        result["dice"]
                    )

                if result["ssim_score"] is not None:
                    ssim_scores.append(
                        result["ssim_score"]
                    )

            data["batch_results"] = batch_results
            data["batch_count"] = len(batch_results)

            if dice_scores:
                data["average_dice"] = round(
                    float(np.mean(dice_scores)),
                    4
                )

            if ssim_scores:
                data["average_ssim"] = round(
                    float(np.mean(ssim_scores)),
                    4
                )

            if batch_results:
                first = batch_results[0]

                for key in (
                    "original_image",
                    "result_image",
                    "overlay_image",
                    "heatmap_image",
                    "inference_time",
                    "dice",
                    "ssim_score"
                ):
                    data[key] = first[key]

            logger.info(
                "Batch completed: %d images | "
                "Average Dice: %s | Average SSIM: %s",
                data["batch_count"],
                data["average_dice"],
                data["average_ssim"]
            )

        except Exception as e:
            logger.exception(
                "Retinal segmentation request failed"
            )

            data["error"] = str(e)

        finally:
            gc.collect()

    return render_template(
        "index.html",
        **data
    )


# ============================================================
# UPLOAD ERROR HANDLER
# ============================================================

@app.errorhandler(413)
def upload_too_large(error):
    return render_template(
        "index.html",
        original_image=None,
        result_image=None,
        overlay_image=None,
        heatmap_image=None,
        dice=None,
        ssim_score=None,
        inference_time=None,
        threshold=0.80,
        dataset="General",
        device_name=DEVICE_NAME,
        error=(
            f"Upload too large. Maximum total "
            f"upload size is {MAX_UPLOAD_MB} MB."
        ),
        batch_results=[],
        batch_count=0,
        average_dice=None,
        average_ssim=None
    ), 413


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
