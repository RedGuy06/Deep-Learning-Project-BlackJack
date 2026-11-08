import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "resnet18_realcard_androom_combinedface.pth"
TABLE_PATH = BASE / "DeepLearningLookupTables.xlsx"

app = Flask(__name__)
# On Render, set ALLOWED_ORIGIN to your Netlify URL, e.g. https://my-site.netlify.app
CORS(app, origins=os.environ.get("ALLOWED_ORIGIN", "*"))

# ---------- Load everything ONCE at startup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

values = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
ranks = values["ranks"]
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(ranks))
model.load_state_dict(values["model_state_dict"])
model = model.to(device).eval()

# Same transforms used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# All strategy sheets, loaded once: {sheet_name: DataFrame}
TABLES = pd.read_excel(TABLE_PATH, sheet_name=None)

ACTIONS = {
    "H": "Hit",
    "S": "Stand",
    "Dh": "Double if allowed, otherwise Hit",
    "Ds": "Double if allowed, otherwise stand",
    "P": "Split",
    "Ph": "Split if double after split is allowed, otherwise Hit",
    "Pd": "Split if double after split is allowed, otherwise double and hit",
    "Rh": "Surrender if allowed, otherwise Hit",
    "Rs": "Surrender if allowed, otherwise Stand",
}

CARD_W, CARD_H = 224, 320
TARGET = np.float32([[0, 0], [CARD_W - 1, 0],
                     [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]])


def hand_total(hand):
    total = sum(1 if r == "A" else int(r) for r in hand)  # aces as 1 first
    if "A" in hand and total + 10 <= 21:
        total += 10  # promote one ace to 11 if it doesn't bust
    return total


def find_cards(image):
    """Return (list of cropped PIL cards, list of 'Dealer'/'Player')."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
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


def recommend(player_hand, dealer_card):
    player_value = hand_total(player_hand)
    minimum = sum(1 if c == "A" else int(c) for c in player_hand)

    sheet = "Total Dependent - Soft" if player_value > minimum else "Total Dependent - Hard"
    if len(player_hand) == 2 and player_hand[0] == player_hand[1]:
        sheet = "Total Dependent - Split"

    df = TABLES[sheet]
    column = "A" if dealer_card == "A" else int(dealer_card)
    try:
        code = df.set_index("PlayerCard")[column].to_dict().get(player_value)
    except KeyError:
        return "Action not found"
    return ACTIONS.get(code, "Action not found")


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

    with Image.open(request.files["image"].stream) as photo:
        image = np.array(ImageOps.exif_transpose(photo).convert("RGB"))

    cards, owners = find_cards(image)

    player_hand, dealer_hand = [], []
    with torch.no_grad():
        for card, owner in zip(cards, owners):
            inputs = transform(card).unsqueeze(0).to(device)
            rank = ranks[model(inputs).argmax(dim=1).item()]
            (dealer_hand if owner == "Dealer" else player_hand).append(rank)

    if not dealer_hand:
        return jsonify(error="Couldn't find a dealer card in the top half of the photo."), 422
    if not player_hand:
        return jsonify(error="Couldn't find any player cards in the bottom half of the photo."), 422

    player_value = hand_total(player_hand)
    dealer_value = hand_total(dealer_hand)

    action = "Bust" if player_value > 21 else recommend(player_hand, dealer_hand[0])

    return jsonify(player_total=player_value, dealer_total=dealer_value, action=action)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
