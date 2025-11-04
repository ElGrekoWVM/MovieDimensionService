# -*- coding: utf-8 -*-
"""Utility functions for calculating visible video aspect ratio and querying Plex.

This module now uses OpenCV (cv2) to detect black bars and compute the visible
aspect ratio of a movie frame sequence instead of relying on ffmpeg cropdetect.

Logs are written to stdout so the Synology Container Manager (container logs)
displays them.
"""
import requests
import os
import cv2
import logging
import sys

# Valid aspect ratio ranges
VALID_RATIOS = [
    (1.75, 1.79),
    (2.35, 2.40),
    (2.20, 2.25),
    (1.84, 1.90)
]

DEFAULT_RATIO = 1.76

# Configure module logger to write to stdout (visible in container manager logs)
logger = logging.getLogger("movie_dimension")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [movie_dimension] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _log(log_func, msg, level="ERROR"):
    if log_func:
        try:
            log_func(msg, level=level)
        except TypeError:
            log_func(msg)
        return

    # fallback to module logger writing to stdout/stderr
    lvl = level.upper() if isinstance(level, str) else level
    if isinstance(lvl, str):
        level_no = getattr(logging, lvl, logging.ERROR)
    else:
        level_no = lvl
    logger.log(level_no, msg)


def calculate_visible_height(file_path, log_func=None, sample_frames=8, black_thresh=16):
    """Use OpenCV to sample frames and detect black bars to compute visible aspect.

    This implementation does NOT rely on the stored frame height metadata for the
    visible height calculation. Instead it derives frame dimensions from actual
    decoded frames and computes the visible bounding box of non-black pixels.

    Args:
      file_path: path to video file
      log_func: optional logging callable
      sample_frames: number of frames to sample across the video
      black_thresh: pixel brightness threshold (0-255) to consider as black

    Returns:
      aspect ratio (rounded to 2 decimals) when a valid ratio is detected
      DEFAULT_RATIO on error or when no valid ratio found
    """
    try:
        if not file_path or not os.path.exists(file_path):
            _log(log_func, f"File does not exist: {file_path}", level="ERROR")
            return DEFAULT_RATIO

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            _log(log_func, f"Could not open video file with OpenCV: {file_path}", level="ERROR")
            return DEFAULT_RATIO

        # Try to get frame count; it may be unreliable. We'll fallback to sequential
        # reading if frame_count is not available.
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # We'll determine frame width/height from the first successfully decoded frame
        width = 0
        height = 0

        # prepare indices when frame_count looks valid
        use_indexed = frame_count > 0
        indices = []
        if use_indexed:
            if frame_count <= sample_frames:
                indices = list(range(frame_count))
            else:
                step = max(1, frame_count // sample_frames)
                indices = [min(frame_count - 1, i * step) for i in range(sample_frames)]

        # accumulator for bounding box of visible (non-black) pixels across sampled frames
        min_x = None
        min_y = None
        max_x = None
        max_y = None
        found_any = False

        if use_indexed and indices:
            _log(log_func, f"Sampling frames by indices: {indices}", level="INFO")
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    _log(log_func, f"Frame read failed at index {idx}", level="WARNING")
                    continue

                if width == 0 or height == 0:
                    height, width = frame.shape[:2]
                    # initialize accumulators
                    min_x, min_y = width, height
                    max_x, max_y = 0, 0

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY)

                coords = cv2.findNonZero(mask)
                if coords is None:
                    _log(log_func, f"No non-black pixels in frame index {idx}", level="DEBUG")
                    continue

                found_any = True
                x_vals = coords[:, 0, 0]
                y_vals = coords[:, 0, 1]
                min_x = min(min_x, int(x_vals.min()))
                max_x = max(max_x, int(x_vals.max()))
                min_y = min(min_y, int(y_vals.min()))
                max_y = max(max_y, int(y_vals.max()))
        else:
            # fallback: read sequential frames until we have enough samples
            _log(log_func, "Frame count not available, using sequential sampling", level="INFO")
            samples_taken = 0
            while samples_taken < sample_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    _log(log_func, "Sequential frame read failed or EOF reached", level="WARNING")
                    break

                if width == 0 or height == 0:
                    height, width = frame.shape[:2]
                    min_x, min_y = width, height
                    max_x, max_y = 0, 0

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY)

                coords = cv2.findNonZero(mask)
                if coords is None:
                    samples_taken += 1
                    continue

                found_any = True
                x_vals = coords[:, 0, 0]
                y_vals = coords[:, 0, 1]
                min_x = min(min_x, int(x_vals.min()))
                max_x = max(max_x, int(x_vals.max()))
                min_y = min(min_y, int(y_vals.min()))
                max_y = max(max_y, int(y_vals.max()))
                samples_taken += 1

        cap.release()

        if not found_any or min_x is None:
            _log(log_func, f"No visible (non-black) pixels detected in sampled frames for {file_path}", level="WARNING")
            return DEFAULT_RATIO

        vis_w = max_x - min_x + 1
        vis_h = max_y - min_y + 1

        if vis_h <= 0 or vis_w <= 0:
            _log(log_func, f"Computed invalid visible dimensions ({vis_w}x{vis_h}) for {file_path}", level="ERROR")
            return DEFAULT_RATIO

        aspect_ratio = round(float(vis_w) / float(vis_h), 2)
        _log(log_func, f"Computed visible dims {vis_w}x{vis_h} -> aspect_ratio={aspect_ratio}", level="INFO")

        # Check if aspect ratio is within valid ranges
        for min_ratio, max_ratio in VALID_RATIOS:
            if min_ratio <= aspect_ratio <= max_ratio:
                _log(log_func, f"Aspect ratio {aspect_ratio} within valid ranges", level="INFO")
                return aspect_ratio

        _log(log_func, f"Aspect ratio {aspect_ratio} not in valid ranges, returning DEFAULT_RATIO", level="WARNING")
        return DEFAULT_RATIO

    except Exception as e:
        _log(log_func, f"Error processing file '{file_path}' with OpenCV: {e}", level="ERROR")
        return DEFAULT_RATIO


def get_file_path_from_plex(rating_key, plex_base_url, plex_token, log_func=None):
    """Query Plex API to retrieve the full file path for a given ratingKey.

    Returns the file path string on success or None on failure.
    """
    try:
        if not plex_base_url or not plex_token:
            _log(log_func, "Plex base URL or token not configured", level="ERROR")
            return None

        url = plex_base_url.rstrip('/') + f"/library/metadata/{rating_key}?includeParts=1"
        headers = {
            'X-Plex-Token': plex_token,
            'Accept': 'application/json'
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # JSON path: MediaContainer -> Metadata[0] -> Media[0] -> Part[0] -> file
        parts = (
            data.get('MediaContainer', {})
                .get('Metadata', [{}])[0]
                .get('Media', [{}])[0]
                .get('Part', [{}])
        )
        if not parts or not parts[0].get('file'):
            _log(log_func, f"No file path in Plex response for ratingKey={rating_key}", level="ERROR")
            return None
        file_path = parts[0]['file']
        _log(log_func, f"Retrieved file path from Plex: {file_path}", level="INFO")
        return file_path
    except Exception as e:
        _log(log_func, f"Error querying Plex API: {e}", level="ERROR")
        return None
