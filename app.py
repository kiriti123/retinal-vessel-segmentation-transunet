
import gc
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    render_template_string,
    request,
    url_for,
)

from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import functional as TF
from werkzeug.exceptions import RequestEntityTooLarge

from model import TransUNetV2


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("retina")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)


# ============================================================
# FLASK SETUP
# ============================================================

BASE = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE / "static" / "uploads"
RESULT_FOLDER = BASE / "static" / "results"
INCOMING_FOLDER = BASE / "incoming_jobs"

for folder in (
    UPLOAD_FOLDER,
    RESULT_FOLDER,
    INCOMING_FOLDER,
):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# Allow all 45 images and their ground-truth masks.
app.config["MAX_CONTENT_LENGTH"] = 350 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 2 * 1024 * 1024

DEVICE_NAME = "CPU"
MAX_IMAGES = 45
MAX_PIXELS = 25_000_000

Image.MAX_IMAGE_PIXELS = MAX_PIXELS

ALLOWED = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}


# ============================================================
# LOAD TRAINED TRANSUNET V2
# ============================================================

model = TransUNetV2()

weights = torch.load(
    str(BASE / "combined_transunet_v2.pt"),
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    weights,
    strict=True,
)

del weights
gc.collect()

model.eval()

log.info("Trained TransUNet V2 loaded on CPU")


# ============================================================
# BACKGROUND JOB STORAGE
# ============================================================

# Jobs exist in the current Gunicorn worker's memory.
# Run Gunicorn with exactly ONE worker.

jobs = {}

jobs_lock = threading.Lock()

# Prevent two batches from running simultaneously.
job_slot = threading.Lock()


# ============================================================
# FILENAME MATCHING
# ============================================================

def key_for(filename):
    stem = Path(filename).stem.lower().strip()

    suffixes = (
        "_manual1",
        "_manual",
        "_ground_truth",
        "_groundtruth",
        "_gt",
        "_mask",
    )

    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]

    return stem


def valid_file(filename):
    return Path(filename).suffix.lower() in ALLOWED


# ============================================================
# LOAD AND RESIZE IMAGE
# ============================================================

def read_resized(
    path,
    mode="RGB",
    resample=Image.Resampling.BILINEAR,
):
    with Image.open(path) as source:

        if source.width * source.height > MAX_PIXELS:
            raise ValueError(
                f"Image too large: {path.name}"
            )

        image = ImageOps.exif_transpose(source).convert(mode)

        resized = image.resize(
            (512, 512),
            resample,
        )

        image.close()

    return resized


# ============================================================
# RETINAL IMAGE PREPROCESSING
# ============================================================

def preprocess(image):
    arr = np.asarray(
        image,
        dtype=np.uint8,
    )

    green = arr[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
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

    enhanced = Image.fromarray(
        np.clip(
            output,
            0,
            255,
        ).astype(np.uint8)
    )

    tensor = TF.to_tensor(enhanced)

    tensor = TF.normalize(
        tensor,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )

    enhanced.close()

    return tensor.unsqueeze(0)


# ============================================================
# MODEL INFERENCE
# ============================================================

def segment(image, threshold):
    tensor = preprocess(image)

    started = time.perf_counter()

    with torch.inference_mode():
        logits = model(tensor)

        probs = (
            torch.sigmoid(logits)[0, 0]
            .cpu()
            .numpy()
            .copy()
        )

    elapsed = round(
        (time.perf_counter() - started) * 1000,
        1,
    )

    del tensor
    del logits

    mask = (
        probs >= threshold
    ).astype(np.uint8) * 255

    heatmap = cv2.applyColorMap(
        np.clip(
            probs * 255,
            0,
            255,
        ).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = np.asarray(image).copy()

    vessel = mask > 0

    overlay[vessel] = (
        0.45 * overlay[vessel]
        + 0.55 * np.array([0, 255, 255])
    ).astype(np.uint8)

    del probs
    del vessel

    return (
        Image.fromarray(mask),
        Image.fromarray(heatmap),
        Image.fromarray(overlay),
        elapsed,
    )


# ============================================================
# DICE AND SSIM
# ============================================================

def metrics(mask_image, gt_path):
    gt = read_resized(
        gt_path,
        "L",
        Image.Resampling.NEAREST,
    )

    try:
        truth = (
            np.asarray(gt) > 127
        ).astype(np.uint8) * 255

        pred = np.asarray(mask_image)

        pred_binary = pred > 0
        truth_binary = truth > 0

        intersection = np.logical_and(
            pred_binary,
            truth_binary,
        ).sum()

        denominator = int(
            pred_binary.sum()
            + truth_binary.sum()
        )

        if denominator == 0:
            dice = 1.0
        else:
            dice = (
                2.0 * intersection / denominator
            )

        structural = ssim(
            pred,
            truth,
            data_range=255,
        )

        return (
            round(float(dice), 4),
            round(float(structural), 4),
        )

    finally:
        gt.close()


# ============================================================
# PROCESS ONE RETINAL IMAGE
# ============================================================

def process_image(image_info, gt_path, threshold):
    image = None
    mask = None
    heat = None
    overlay = None

    try:
        image = read_resized(
            Path(image_info["path"])
        )

        (
            mask,
            heat,
            overlay,
            elapsed,
        ) = segment(image, threshold)

        prefix = uuid.uuid4().hex[:12]

        original_name = prefix + "_original.png"
        mask_name = prefix + "_mask.png"
        overlay_name = prefix + "_overlay.png"
        heat_name = prefix + "_heatmap.png"

        image.save(
            UPLOAD_FOLDER / original_name
        )

        mask.save(
            RESULT_FOLDER / mask_name
        )

        overlay.save(
            RESULT_FOLDER / overlay_name
        )

        heat.save(
            RESULT_FOLDER / heat_name
        )

        dice = None
        structural = None

        if gt_path:
            dice, structural = metrics(
                mask,
                Path(gt_path["path"]),
            )

        return {
            "filename": image_info["filename"],
            "ground_truth": (
                gt_path["filename"]
                if gt_path
                else None
            ),
            "original_image": (
                "uploads/" + original_name
            ),
            "result_image": (
                "results/" + mask_name
            ),
            "overlay_image": (
                "results/" + overlay_name
            ),
            "heatmap_image": (
                "results/" + heat_name
            ),
            "inference_time": elapsed,
            "dice": dice,
            "ssim_score": structural,
        }

    finally:
        for item in (
            image,
            mask,
            heat,
            overlay,
        ):
            if item is not None:
                item.close()

        gc.collect()


# ============================================================
# UPDATE BACKGROUND JOB
# ============================================================

def update_job(job_id, **changes):
    with jobs_lock:
        jobs[job_id].update(changes)


# ============================================================
# BACKGROUND BATCH PROCESSING
# ============================================================

def run_batch(
    job_id,
    images,
    gt_lookup,
    threshold,
    incoming_dir,
):
    try:
        update_job(
            job_id,
            status="running",
        )

        results = []

        for number, info in enumerate(images, 1):

            log.info(
                "[%d/%d] Processing %s",
                number,
                len(images),
                info["filename"],
            )

            gt_path = gt_lookup.get(
                key_for(info["filename"])
            )

            result = process_image(
                info,
                gt_path,
                threshold,
            )

            results.append(result)

            update_job(
                job_id,
                completed=number,
                current=info["filename"],
                results=results.copy(),
            )

            log.info(
                "Completed %d/%d; Dice=%s; SSIM=%s",
                number,
                len(images),
                result["dice"],
                result["ssim_score"],
            )

        dice_scores = [
            r["dice"]
            for r in results
            if r["dice"] is not None
        ]

        ssim_scores = [
            r["ssim_score"]
            for r in results
            if r["ssim_score"] is not None
        ]

        update_job(
            job_id,
            status="completed",
            average_dice=(
                round(
                    float(np.mean(dice_scores)),
                    4,
                )
                if dice_scores
                else None
            ),
            average_ssim=(
                round(
                    float(np.mean(ssim_scores)),
                    4,
                )
                if ssim_scores
                else None
            ),
            current="Completed",
        )

        log.info(
            "Batch %s completed: %d images",
            job_id,
            len(results),
        )

    except Exception as exc:
        log.exception(
            "Batch %s failed",
            job_id,
        )

        update_job(
            job_id,
            status="failed",
            error=str(exc),
        )

    finally:
        shutil.rmtree(
            incoming_dir,
            ignore_errors=True,
        )

        job_slot.release()


# ============================================================
# DEFAULT TEMPLATE DATA
# ============================================================

def default_data():
    return {
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
        "average_ssim": None,
    }


# ============================================================
# PROGRESS PAGE
# ============================================================

PROGRESS_HTML = """
<!doctype html>
<html lang="en">

<head>
<meta charset="utf-8">
<meta name="viewport"
content="width=device-width,initial-scale=1">

<title>Retinal Segmentation - Batch Progress</title>

<style>
body {
    font-family: system-ui, Arial;
    background: #f5f8fc;
    color: #172a44;
    margin: 0;
    padding: 35px 16px;
}

main {
    max-width: 720px;
    margin: 40px auto;
    background: white;
    padding: 32px;
    border-radius: 16px;
    box-shadow: 0 8px 30px #172a4415;
}

progress {
    width: 100%;
    height: 25px;
    accent-color: #1581c4;
}

a {
    color: #126ab0;
}

small {
    color: #56677a;
}
</style>

</head>

<body>

<main>

<h1>Retinal Vessel Segmentation</h1>

<h2 id="heading">Preparing batch...</h2>

<progress
    id="bar"
    value="0"
    max="{{ total }}">
</progress>

<p id="count">
0 / {{ total }} images completed
</p>

<p id="current"></p>

<p id="message">
Please keep this page open to watch progress.
</p>

<p>
<a href="{{ url_for('index') }}">
Return to upload page
</a>
</p>

<small>
Processing runs on the server.
Results remain available while the server is running.
</small>

</main>

<script>

const endpoint = {{
    url_for('batch_status', job_id=job_id)|tojson
}};

const results = {{
    url_for('batch_results', job_id=job_id)|tojson
}};

async function poll() {
    try {
        const response = await fetch(
            endpoint,
            {cache: 'no-store'}
        );

        if (!response.ok) {
            throw new Error(
                'Status unavailable (' + response.status + ')'
            );
        }

        const job = await response.json();

        document.getElementById('bar').value =
            job.completed;

        document.getElementById('count').textContent =
            job.completed + ' / ' +
            job.total + ' images completed';

        document.getElementById('current').textContent =
            job.current || '';

        if (job.status === 'completed') {
            document.getElementById('heading').textContent =
                'Batch complete!';

            document.getElementById('message').innerHTML =
                '<a href="' + results +
                '">View all results, Dice and SSIM</a>';

            window.location.replace(results);
            return;
        }

        if (job.status === 'failed') {
            document.getElementById('heading').textContent =
                'Batch stopped';

            document.getElementById('message').textContent =
                job.error || 'An error occurred';

            return;
        }

        if (job.status === 'running') {
            document.getElementById('heading').textContent =
                'Processing retinal images...';
        }

    } catch (err) {
        document.getElementById('message').textContent =
            'Unable to reach server: ' +
            err.message + ' — try refreshing.';
    }

    setTimeout(poll, 2000);
}

poll();

</script>

</body>
</html>
"""


# ============================================================
# MAIN WEBSITE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "GET":
        return render_template(
            "index.html",
            **default_data(),
        )

    data = default_data()

    if not job_slot.acquire(blocking=False):
        data["error"] = (
            "Another batch is processing. "
            "Please wait until it finishes."
        )

        return render_template(
            "index.html",
            **data,
        ), 429

    incoming_dir = None
    started = False

    try:
        files = [
            f
            for f in request.files.getlist("image")
            if f and f.filename
        ]

        gt_files = [
            f
            for f in request.files.getlist("ground_truth")
            if f and f.filename
        ]

        if not files:
            raise ValueError(
                "Please select retinal images."
            )

        if len(files) > MAX_IMAGES:
            raise ValueError(
                f"Maximum {MAX_IMAGES} fundus images per upload."
            )

        if len(gt_files) > MAX_IMAGES:
            raise ValueError(
                f"Maximum {MAX_IMAGES} ground-truth masks per upload."
            )

        if any(
            not valid_file(f.filename)
            for f in files + gt_files
        ):
            raise ValueError(
                "Allowed formats: JPG, JPEG, PNG, TIFF, BMP."
            )

        threshold = float(
            request.form.get(
                "threshold",
                ".80",
            )
        )

        if (
            not np.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        dataset = request.form.get(
            "dataset",
            "General",
        )

        job_id = uuid.uuid4().hex

        incoming_dir = (
            INCOMING_FOLDER / job_id
        )

        incoming_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        images = []
        gt_lookup = {}

        # Save incoming files before starting the thread.
        for f in files:
            dest = incoming_dir / (
                uuid.uuid4().hex
                + Path(f.filename).suffix.lower()
            )

            f.save(dest)

            images.append({
                "filename": Path(f.filename).name,
                "path": str(dest),
            })

        for f in gt_files:
            dest = incoming_dir / (
                uuid.uuid4().hex
                + Path(f.filename).suffix.lower()
            )

            f.save(dest)

            gt_lookup[
                key_for(f.filename)
            ] = {
                "path": str(dest),
                "filename": Path(f.filename).name,
            }

        with jobs_lock:
            jobs[job_id] = {
                "status": "queued",
                "total": len(images),
                "completed": 0,
                "current": "",
                "error": None,
                "results": [],
                "threshold": threshold,
                "dataset": dataset,
                "average_dice": None,
                "average_ssim": None,
            }

        thread = threading.Thread(
            target=run_batch,
            args=(
                job_id,
                images,
                gt_lookup,
                threshold,
                incoming_dir,
            ),
            daemon=True,
            name="retina-batch-" + job_id[:8],
        )

        thread.start()
        started = True

        return redirect(
            url_for(
                "batch_progress",
                job_id=job_id,
            ),
            code=303,
        )

    except Exception as exc:
        log.exception("Upload failed")

        data["error"] = str(exc)

        return render_template(
            "index.html",
            **data,
        ), 400

    finally:
        if not started:
            if incoming_dir is not None:
                shutil.rmtree(
                    incoming_dir,
                    ignore_errors=True,
                )

            job_slot.release()


# ============================================================
# BATCH PROGRESS PAGE
# ============================================================

@app.get("/batch/<job_id>")
def batch_progress(job_id):

    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            return (
                "Job not found. Server may have restarted.",
                404,
            )

        total = job["total"]

    return render_template_string(
        PROGRESS_HTML,
        job_id=job_id,
        total=total,
    )


# ============================================================
# BATCH STATUS API
# ============================================================

@app.get("/batch/<job_id>/status")
def batch_status(job_id):

    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            return jsonify(
                error="Job not found; server may have restarted"
            ), 404

        return jsonify({
            "status": job["status"],
            "total": job["total"],
            "completed": job["completed"],
            "current": job["current"],
            "error": job["error"],
        })


# ============================================================
# FINAL BATCH RESULTS
# ============================================================

@app.get("/batch/<job_id>/results")
def batch_results(job_id):

    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            return (
                "Job not found. Server may have restarted.",
                404,
            )

        snapshot = dict(job)

        snapshot["results"] = (
            job["results"].copy()
        )

    if snapshot["status"] not in (
        "completed",
        "failed",
    ):
        return redirect(
            url_for(
                "batch_progress",
                job_id=job_id,
            )
        )

    data = default_data()

    data.update({
        "threshold": snapshot["threshold"],
        "dataset": snapshot["dataset"],
        "batch_results": snapshot["results"],
        "batch_count": len(snapshot["results"]),
        "average_dice": snapshot["average_dice"],
        "average_ssim": snapshot["average_ssim"],
        "error": snapshot["error"],
    })

    if snapshot["results"]:
        first = snapshot["results"][0]

        for key in (
            "original_image",
            "result_image",
            "overlay_image",
            "heatmap_image",
            "dice",
            "ssim_score",
            "inference_time",
        ):
            data[key] = first[key]

    return render_template(
        "index.html",
        **data,
    )


# ============================================================
# UPLOAD SIZE ERROR
# ============================================================

@app.errorhandler(RequestEntityTooLarge)
def too_large(error):

    data = default_data()

    data["error"] = (
        "Upload exceeds 350 MB total. "
        "Please reduce image sizes."
    )

    return render_template(
        "index.html",
        **data,
    ), 413


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
    )
