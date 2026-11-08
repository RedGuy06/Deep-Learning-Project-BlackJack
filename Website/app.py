import os
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

import actionLookup
import lookupActionFromTable

BASE = Path(__file__).resolve().parent


def find_model():
    """Use the .pth file in Website/models/, whatever it's called."""
    folder = BASE / "models"
    wanted = os.environ.get("MODEL_FILE")  # optional: choose a file by name
    if wanted:
        path = folder / wanted
        if not path.is_file():
            raise FileNotFoundError(f"MODEL_FILE={wanted} was not found in {folder}")
        return path
    found = list(folder.glob("*.pth"))
    if not found:
        raise FileNotFoundError(f"No .pth model file found in {folder}")
    # Newest by modified time (file name breaks a tie).
    newest = max(found, key=lambda p: (p.stat().st_mtime, p.name))
    if len(found) > 1:
        print(f"Several .pth files in {folder}; using the newest, {newest.name}. "
              "Set MODEL_FILE to choose another.", flush=True)
    return newest


MODEL_PATH = find_model()
print(f"Loading model: {MODEL_PATH.name}", flush=True)

app = Flask(__name__)
# On the host, set ALLOWED_ORIGIN to your Netlify URL, e.g. https://my-site.netlify.app
CORS(app, origins=os.environ.get("ALLOWED_ORIGIN", "*"))

# ---------- Load everything ONCE at startup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

values = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
ranks = values["ranks"]
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(ranks))
model.load_state_dict(values["model_state_dict"])
model = model.to(device).eval()
torch.set_num_threads(1)  # less memory/CPU contention on a small instance

# Same transforms used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Strategy sheets, loaded once: {sheet_name: DataFrame}
TABLES = lookupActionFromTable.load_tables()

MAX_SIDE = 1600  # longest image side processed; bigger photos are shrunk

CARD_W, CARD_H = 224, 320
TARGET = np.float32([[0, 0], [CARD_W - 1, 0],
                     [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]])


def find_cards(image):
    """Return (list of cropped PIL cards, list of 'Dealer'/'Player')."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    split_y = image.shape[0] / 2
    min_area = 0.01 * image.shape[0] * image.shape[1]
    cards, owners = [], []

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        points = cv2.boxPoints(cv2.minAreaRect(contour))
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        points = points[np.argsort(angles)]

        sides = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
        if sides[0] + sides[2] > sides[1] + sides[3]:
            points = np.roll(points, -1, axis=0)

        matrix = cv2.getPerspectiveTransform(points, TARGET)
        crop = cv2.warpPerspective(image, matrix, (CARD_W, CARD_H))
        cards.append(Image.fromarray(crop))
        owners.append("Dealer" if center[1] < split_y else "Player")

    return cards, owners


def parse_count():
    """Running count sent by the page, or None if the page isn't counting."""
    raw = request.form.get("count")
    if raw is None:
        return None
    try:
        return max(-500, min(500, int(raw)))
    except ValueError:
        return None


@app.get("/")
def home():
    return "Blackjack Advisor API is running."


@app.get("/health")
def health():
    return "ok"


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(error="No image was uploaded."), 400

    # Decode at reduced size to keep memory low (big phone photos are huge raw).
    with Image.open(request.files["image"].stream) as raw:
        raw.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEG: decode smaller from the start
        photo = ImageOps.exif_transpose(raw)
    photo.thumbnail((MAX_SIDE, MAX_SIDE))
    image = np.array(photo.convert("RGB"))
    del photo

    cards, owners = find_cards(image)

    player_ranks, dealer_ranks = [], []
    with torch.no_grad():
        for card, owner in zip(cards, owners):
            inputs = transform(card).unsqueeze(0).to(device)
            rank = ranks[model(inputs).argmax(dim=1).item()]
            (dealer_ranks if owner == "Dealer" else player_ranks).append(rank)

    try:
        result = actionLookup.recommend(player_ranks, dealer_ranks, parse_count(), TABLES)
    except ValueError as err:
        return jsonify(error=str(err)), 422

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
