# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import cv2
import numpy as np
import logging

app = Flask(__name__)

# Konfiguriere Logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Gültige Aspect-Ratio-Bereiche
VALID_RATIOS = [
    (1.75, 1.79),
    (2.35, 2.40),
    (2.20, 2.25),
    (1.84, 1.90)
]

DEFAULT_RATIO = 1.76

def calculate_visible_height(file_path):
    try:
        # Video laden
        video = cv2.VideoCapture(file_path)
        if not video.isOpened():
            raise ValueError("Video konnte nicht geoeffnet werden.")

        # Gesamtanzahl der Frames abrufen
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_index = total_frames // 2

        # Zum mittleren Frame springen
        video.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        success, frame = video.read()
        if not success:
            raise ValueError("Mittlerer Frame konnte nicht gelesen werden.")

        # Frame in Graustufen konvertieren
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Schwarze Balken erkennen (Pixel mit sehr niedrigen Werten)
        threshold = 10  # Schwellenwert für Schwarz
        mask = gray_frame > threshold

        # Sichtbare Höhe berechnen
        visible_rows = np.any(mask, axis=1)
        visible_height = np.sum(visible_rows)

        # Breite des Frames abrufen
        frame_width = frame.shape[1]

        # Video freigeben
        video.release()

        # Aspect Ratio berechnen
        aspect_ratio = round(frame_width / visible_height, 2)

        # Prüfen, ob die Aspect Ratio in den gültigen Bereichen liegt
        for min_ratio, max_ratio in VALID_RATIOS:
            if min_ratio <= aspect_ratio <= max_ratio:
                return aspect_ratio

        # Wenn keine gültige Ratio gefunden wurde, Standardwert zurückgeben
        return DEFAULT_RATIO
    except Exception as e:
        # Fehler ins Docker-Log schreiben
        logging.error(f"Fehler beim Verarbeiten der Datei '{file_path}': {e}")
        return DEFAULT_RATIO

@app.route('/aspect-ratio', methods=['POST'])
def get_aspect_ratio():
    try:
        data = request.get_json(force=True)  # Force parsing if Content-Type is incorrect
        if not data or 'file_path' not in data:
            return jsonify({"error": "Invalid or missing 'file_path' in JSON payload."}), 400

        file_path = data['file_path']  # Extract file_path from JSON payload

        if not file_path:
            return jsonify({"error": "Kein Dateipfad angegeben."}), 400

        aspect_ratio = calculate_visible_height(file_path)
        return jsonify({"aspect_ratio": aspect_ratio})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
