import os
import time
import uuid

import cv2
import numpy as np
import torch

from flask import Flask, render_template, request
from PIL import Image
from torchvision.transforms import functional as TF
from skimage.metrics import structural_similarity as ssim

from model import TransUNetV2

# Optimize PyTorch CPU performance for Render
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ============================================================
# FLASK SETUP
# ============================================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================
# CPU is intentionally used for the Flask demo.
# This avoids Apple MPS / Metal crashes during large batch runs.
# ============================================================

device = torch.device("cpu")
DEVICE_NAME = "CPU"

print("Using device:", device)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "combined_transunet_v2.pt"

model = TransUNetV2()

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(
    state_dict,
    strict=True
)

model = model.to(device)
model.eval()

print("✅ Trained TransUNet model loaded successfully")


# ============================================================
# RETINAL IMAGE ENHANCEMENT
# ============================================================

def enhance_retina_rgb(img_pil):

    arr = np.array(
        img_pil.convert("RGB")
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
        output,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(output)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (512, 512),
        Image.Resampling.BILINEAR
    )

    image = enhance_retina_rgb(image)

    tensor = TF.to_tensor(image)

    tensor = TF.normalize(
        tensor,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

    return tensor.unsqueeze(0)


# ============================================================
# DICE SCORE
# ============================================================

def calculate_dice(pred, truth):

    pred = pred > 0
    truth = truth > 0

    intersection = np.logical_and(
        pred,
        truth
    ).sum()

    denominator = (
        pred.sum()
        + truth.sum()
    )

    if denominator == 0:
        return 1.0

    return (
        2.0
        * intersection
        / denominator
    )


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(pred, truth):

    pred = pred.astype(np.uint8)
    truth = truth.astype(np.uint8)

    return ssim(
        pred,
        truth,
        data_range=255
    )


# ============================================================
# NORMALIZE FILENAMES
# ============================================================

def get_file_key(filename):

    filename = os.path.basename(filename)

    name, _ = os.path.splitext(filename)

    name = name.lower().strip()

    # Remove common GT suffixes if they exist

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

    return name


# ============================================================
# BUILD GROUND-TRUTH LOOKUP
# ============================================================

def build_ground_truth_lookup(files):

    lookup = {}

    for file in files:

        if not file:
            continue

        if not file.filename:
            continue

        key = get_file_key(
            file.filename
        )

        lookup[key] = file

    return lookup


# ============================================================
# MODEL INFERENCE
# ============================================================

def segment_retina(image, threshold):

    original_size = image.size

    tensor = preprocess_image(
        image
    ).to(device)


    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    start = time.perf_counter()

    with torch.inference_mode():

        logits = model(tensor)

        probabilities = torch.sigmoid(
            logits
        )

    inference_time = (
        time.perf_counter()
        - start
    ) * 1000


    # --------------------------------------------------------
    # PROBABILITY MAP
    # --------------------------------------------------------

    probability_map = (
        probabilities
        .squeeze()
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # BINARY VESSEL MASK
    # --------------------------------------------------------

    mask = (
        probability_map >= threshold
    ).astype(np.uint8) * 255


    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    probability_uint8 = (
        probability_map * 255
    ).clip(
        0,
        255
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        probability_uint8,
        cv2.COLORMAP_TURBO
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # RESIZE OUTPUT TO ORIGINAL SIZE
    # --------------------------------------------------------

    mask_image = Image.fromarray(mask)

    mask_image = mask_image.resize(
        original_size,
        Image.Resampling.NEAREST
    )


    heatmap_image = Image.fromarray(
        heatmap
    )

    heatmap_image = heatmap_image.resize(
        original_size,
        Image.Resampling.BILINEAR
    )


    # --------------------------------------------------------
    # CREATE VESSEL OVERLAY
    # --------------------------------------------------------

    original_np = np.array(
        image.convert("RGB")
    )

    resized_mask = (
        np.array(mask_image) > 0
    )

    overlay = original_np.copy()

    vessel_layer = np.zeros_like(
        overlay
    )

    # Cyan vessel visualization

    vessel_layer[:, :, 0] = 0
    vessel_layer[:, :, 1] = 255
    vessel_layer[:, :, 2] = 255

    overlay[resized_mask] = (
        0.45 * overlay[resized_mask]
        + 0.55 * vessel_layer[resized_mask]
    ).astype(np.uint8)

    overlay_image = Image.fromarray(
        overlay
    )


    return (
        mask_image,
        heatmap_image,
        overlay_image,
        inference_time
    )


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


    # ========================================================
    # POST REQUEST
    # ========================================================

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # FUNDUS IMAGES
            # ------------------------------------------------

            files = request.files.getlist(
                "image"
            )

            files = [
                file
                for file in files
                if file and file.filename
            ]


            if not files:

                raise ValueError(
                    "Please select at least one retinal image."
                )


            # ------------------------------------------------
            # DATASET
            # ------------------------------------------------

            dataset = request.form.get(
                "dataset",
                "General"
            )

            data["dataset"] = dataset


            # ------------------------------------------------
            # THRESHOLD
            # ------------------------------------------------

            threshold = float(
                request.form.get(
                    "threshold",
                    0.80
                )
            )

            threshold = max(
                0.0,
                min(
                    1.0,
                    threshold
                )
            )

            data["threshold"] = threshold


            # ------------------------------------------------
            # GROUND TRUTH FILES
            # ------------------------------------------------

            ground_truth_files = (
                request.files.getlist(
                    "ground_truth"
                )
            )

            ground_truth_files = [
                file
                for file in ground_truth_files
                if file and file.filename
            ]


            # ------------------------------------------------
            # BUILD GT LOOKUP
            # ------------------------------------------------

            gt_lookup = (
                build_ground_truth_lookup(
                    ground_truth_files
                )
            )


            # =================================================
            # DEBUG INFORMATION
            # =================================================

            print("\n")
            print("=" * 60)
            print("RETINAL SEGMENTATION BATCH")
            print("=" * 60)

            print(
                "Fundus images:",
                len(files)
            )

            print(
                "Ground truth masks:",
                len(ground_truth_files)
            )


            # ------------------------------------------------
            # PRINT FIRST IMAGE FILENAMES
            # ------------------------------------------------

            print(
                "\n--- FIRST FUNDUS FILENAMES ---"
            )

            for image_file in files[:10]:

                print(
                    image_file.filename,
                    "-> key:",
                    get_file_key(
                        image_file.filename
                    )
                )


            # ------------------------------------------------
            # PRINT FIRST GT FILENAMES
            # ------------------------------------------------

            print(
                "\n--- FIRST GROUND TRUTH FILENAMES ---"
            )

            for gt_file in ground_truth_files[:10]:

                print(
                    gt_file.filename,
                    "-> key:",
                    get_file_key(
                        gt_file.filename
                    )
                )

            print(
                "------------------------------------"
            )


            # =================================================
            # RESULT STORAGE
            # =================================================

            batch_results = []

            dice_scores = []

            ssim_scores = []

            matched_count = 0


            # =================================================
            # PROCESS IMAGES ONE AT A TIME
            # =================================================

            for number, file in enumerate(
                files,
                start=1
            ):

                original_filename = (
                    file.filename
                )

                image_key = get_file_key(
                    original_filename
                )


                print(
                    f"\n[{number}/{len(files)}]",
                    "Processing:",
                    original_filename
                )


                # ------------------------------------------------
                # LOAD FUNDUS IMAGE
                # ------------------------------------------------

                file.stream.seek(0)

                image = Image.open(
                    file.stream
                ).convert("RGB")


                # ------------------------------------------------
                # OUTPUT FILENAMES
                # ------------------------------------------------

                run_id = (
                    uuid.uuid4()
                    .hex[:10]
                )

                original_name = (
                    f"{run_id}_original.png"
                )

                mask_name = (
                    f"{run_id}_mask.png"
                )

                overlay_name = (
                    f"{run_id}_overlay.png"
                )

                heatmap_name = (
                    f"{run_id}_heatmap.png"
                )


                # ------------------------------------------------
                # SAVE ORIGINAL
                # ------------------------------------------------

                image.save(
                    os.path.join(
                        UPLOAD_FOLDER,
                        original_name
                    )
                )


                # ------------------------------------------------
                # RUN MODEL
                # ------------------------------------------------

                (
                    mask_image,
                    heatmap_image,
                    overlay_image,
                    inference_time

                ) = segment_retina(
                    image,
                    threshold
                )


                # ------------------------------------------------
                # SAVE MODEL OUTPUTS
                # ------------------------------------------------

                mask_image.save(
                    os.path.join(
                        RESULT_FOLDER,
                        mask_name
                    )
                )

                overlay_image.save(
                    os.path.join(
                        RESULT_FOLDER,
                        overlay_name
                    )
                )

                heatmap_image.save(
                    os.path.join(
                        RESULT_FOLDER,
                        heatmap_name
                    )
                )


                # =================================================
                # METRIC DEFAULTS
                # =================================================

                image_dice = None

                image_ssim = None

                matched_gt_name = None


                # =================================================
                # FIND MATCHING GT
                # =================================================

                gt_file = gt_lookup.get(
                    image_key
                )


                if gt_file is not None:

                    matched_count += 1

                    matched_gt_name = (
                        gt_file.filename
                    )


                    print(
                        "   ✅ GT MATCH:",
                        original_filename,
                        "<->",
                        matched_gt_name
                    )


                    # --------------------------------------------
                    # READ GT
                    # --------------------------------------------

                    gt_file.stream.seek(0)

                    gt = Image.open(
                        gt_file.stream
                    ).convert("L")


                    # --------------------------------------------
                    # RESIZE GT
                    # --------------------------------------------

                    gt = gt.resize(
                        mask_image.size,
                        Image.Resampling.NEAREST
                    )


                    # --------------------------------------------
                    # GT ARRAY
                    # --------------------------------------------

                    gt_np = np.array(gt)


                    # --------------------------------------------
                    # BINARY GT
                    # --------------------------------------------

                    gt_np = (
                        gt_np > 127
                    ).astype(
                        np.uint8
                    ) * 255


                    # --------------------------------------------
                    # PREDICTION ARRAY
                    # --------------------------------------------

                    pred_np = np.array(
                        mask_image
                    )


                    # --------------------------------------------
                    # DICE
                    # --------------------------------------------

                    image_dice = round(
                        calculate_dice(
                            pred_np,
                            gt_np
                        ),
                        4
                    )


                    # --------------------------------------------
                    # SSIM
                    # --------------------------------------------

                    image_ssim = round(
                        calculate_ssim(
                            pred_np,
                            gt_np
                        ),
                        4
                    )


                    dice_scores.append(
                        image_dice
                    )

                    ssim_scores.append(
                        image_ssim
                    )


                    print(
                        "   Dice:",
                        image_dice
                    )

                    print(
                        "   SSIM:",
                        image_ssim
                    )


                else:

                    if ground_truth_files:

                        print(
                            "   ⚠️ NO GT MATCH"
                        )

                        print(
                            "   Image key:",
                            image_key
                        )


                # =================================================
                # ADD IMAGE RESULT
                # =================================================

                batch_results.append(
                    {

                        "filename":
                            original_filename,

                        "ground_truth":
                            matched_gt_name,

                        "original_image":
                            "uploads/"
                            + original_name,

                        "result_image":
                            "results/"
                            + mask_name,

                        "overlay_image":
                            "results/"
                            + overlay_name,

                        "heatmap_image":
                            "results/"
                            + heatmap_name,

                        "inference_time":
                            round(
                                inference_time,
                                1
                            ),

                        "dice":
                            image_dice,

                        "ssim_score":
                            image_ssim
                    }
                )


                # Explicitly release image tensors/arrays
                # before processing the next file.

                del image
                del mask_image
                del heatmap_image
                del overlay_image


            # =================================================
            # BATCH RESULTS
            # =================================================

            data["batch_results"] = (
                batch_results
            )

            data["batch_count"] = len(
                batch_results
            )


            # =================================================
            # AVERAGE DICE
            # =================================================

            if dice_scores:

                data["average_dice"] = round(
                    float(
                        np.mean(
                            dice_scores
                        )
                    ),
                    4
                )


            # =================================================
            # AVERAGE SSIM
            # =================================================

            if ssim_scores:

                data["average_ssim"] = round(
                    float(
                        np.mean(
                            ssim_scores
                        )
                    ),
                    4
                )


            # =================================================
            # TERMINAL SUMMARY
            # =================================================

            print("\n")
            print("=" * 60)
            print("BATCH COMPLETE")
            print("=" * 60)

            print(
                "Images processed:",
                len(batch_results)
            )

            print(
                "Ground-truth matches:",
                matched_count,
                "/",
                len(files)
            )


            if dice_scores:

                print(
                    "Average Dice:",
                    data["average_dice"]
                )


            if ssim_scores:

                print(
                    "Average SSIM:",
                    data["average_ssim"]
                )

            print("=" * 60)
            print("\n")


            # =================================================
            # FIRST IMAGE FOR EXISTING COMPARISON SECTION
            # =================================================

            if batch_results:

                first = batch_results[0]

                data["original_image"] = (
                    first[
                        "original_image"
                    ]
                )

                data["result_image"] = (
                    first[
                        "result_image"
                    ]
                )

                data["overlay_image"] = (
                    first[
                        "overlay_image"
                    ]
                )

                data["heatmap_image"] = (
                    first[
                        "heatmap_image"
                    ]
                )

                data["inference_time"] = (
                    first[
                        "inference_time"
                    ]
                )

                data["dice"] = (
                    first[
                        "dice"
                    ]
                )

                data["ssim_score"] = (
                    first[
                        "ssim_score"
                    ]
                )


        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except Exception as e:

            print(
                "\n❌ FLASK ERROR:",
                repr(e)
            )

            data["error"] = str(e)


    # ========================================================
    # LOAD HTML
    # ========================================================

    return render_template(
        "index.html",
        **data
    )


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
