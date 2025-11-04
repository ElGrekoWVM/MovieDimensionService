import cv2
import numpy as np
from appdaemon.plugins.hass.hassapi import Hass

# Gültige Aspect-Ratio-Bereiche
VALID_RATIOS = [
    (1.75, 1.79),
    (2.35, 2.40),
    (2.20, 2.25),
    (1.84, 1.90)
]

DEFAULT_RATIO = 1.76


def calculate_visible_height(file_path, log_func=None):
    """Lädt ein Video, bestimmt die sichtbare Höhe (ohne schwarze Balken)
    und gibt die berechnete Aspect Ratio zurück. Falls ein Fehler auftritt,
    wird DEFAULT_RATIO zurückgegeben. Optional kann ein Log-Callback übergeben
    werden (z.B. self.log von AppDaemon).
    """
    try:
        video = cv2.VideoCapture(file_path)
        if not video.isOpened():
            raise ValueError("Video konnte nicht geöffnet werden")

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        middle_frame_index = max(0, total_frames // 2)

        video.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        success, frame = video.read()
        if not success or frame is None:
            # Try to read the first frame as fallback
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video.read()
            if not success or frame is None:
                video.release()
                raise ValueError("Kein Frame lesbar aus dem Video")

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Schwarze Balken erkennen (Pixel mit sehr niedrigen Werten)
        threshold = 10
        mask = gray_frame > threshold

        # Sichtbare Höhe berechnen
        visible_rows = np.any(mask, axis=1)
        visible_height = int(np.sum(visible_rows))

        frame_width = int(frame.shape[1])
        video.release()

        # Guard against zero division
        if visible_height <= 0:
            aspect_ratio = DEFAULT_RATIO
        else:
            aspect_ratio = round(float(frame_width) / float(visible_height), 2)

        # Prüfen, ob die Aspect Ratio in den gültigen Bereichen liegt
        for min_ratio, max_ratio in VALID_RATIOS:
            if min_ratio <= aspect_ratio <= max_ratio:
                return aspect_ratio

        # Wenn keine gültige Ratio gefunden wurde, Standardwert zurückgeben
        return DEFAULT_RATIO

    except Exception as e:
        if log_func:
            try:
                log_func(f"Fehler beim Verarbeiten der Datei '{file_path}': {e}", level="ERROR")
            except TypeError:
                # some log_func implementations may not accept level kwarg
                log_func(f"Fehler beim Verarbeiten der Datei '{file_path}': {e}")
        return DEFAULT_RATIO


class MovieDimension(Hass):
    """AppDaemon app to calculate the visible aspect ratio of a movie file.

    Diese Version enthält die Kernfunktionalität direkt und registriert keinen
    Home Assistant Service mehr. Verwende interne Methoden oder rufe
    `calculate_and_set(file_path, target_entity)` von anderen Apps aus.
    """

    def initialize(self):
        # Registriere Home Assistant Service: namespace=movie_dimension, service=calculate
        # damit Automationen z.B. service: movie_dimension/calculate aufrufen können.
        self.register_service("movie_dimension", "calculate", self.handle_calculate)
        self.log("MovieDimension App initialisiert und Service movie_dimension/calculate registriert")

    def handle_calculate(self, namespace, data, kwargs):
        # data kommt aus dem Service call. Erwartet: file_path (required), target_entity (optional)
        file_path = data.get("file_path") if data else None
        target_entity = data.get("target_entity") if data else None

        if not file_path:
            self.log("Kein Dateipfad im Service-Call angegeben", level="ERROR")
            return

        self.calculate_and_set(file_path, target_entity)

    def calculate_and_set(self, file_path, target_entity=None):
        """Berechnet die Aspect-Ratio und setzt den State des target_entity in Home Assistant.

        Kann von anderen AppDaemon-Apps oder innerhalb dieser App aufgerufen werden.
        """
        if not target_entity:
            target_entity = "sensor.movie_aspect_ratio"

        aspect = calculate_visible_height(file_path, log_func=self.log)
        valid = aspect != DEFAULT_RATIO

        state = str(aspect)
        attributes = {
            "file_path": file_path,
            "aspect_ratio": aspect,
            "valid": valid
        }

        self.set_state(target_entity, state=state, attributes=attributes)
        self.log(f"Berechnete Aspect-Ratio={aspect} (valid={valid}) fuer {file_path}")
