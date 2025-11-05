# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import os
import re
from movie_dimension import calculate_visible_height, get_file_path_from_plex

app = Flask(__name__)

# In Synology the shared folder is mounted at /video inside the container
VIDEO_ROOT = os.environ.get('VIDEO_ROOT', '/video')


def map_plex_path_to_container(plex_path):
    """Map a Plex filesystem path (Windows or Linux style) to the container VIDEO_ROOT.

    Examples:
      "Y:\\Filme\\...\\file.mkv" -> "/video/Filme/.../file.mkv"
      "/mnt/plex/volume/.../file.mkv" -> returned unchanged if it already contains VIDEO_ROOT
    """
    if not plex_path:
        return None

    # Normalize separators to forward slashes
    path = plex_path.replace('\\\\', '/').replace('\\', '/').strip()

    # If path already points inside the container root, return as-is
    if path.startswith(VIDEO_ROOT):
        return path

    # Windows drive letter, e.g. Y:/...
    m = re.match(r'^[A-Za-z]:/(.*)', path)
    if m:
        rel = m.group(1).lstrip('/')
        return os.path.join(VIDEO_ROOT, rel)

    # UNC path or leading slashes: remove leading // or / and join
    if path.startswith('//') or path.startswith('/'):
        rel = path.lstrip('/')
        return os.path.join(VIDEO_ROOT, rel)

    # Fallback: take the basename and place it under VIDEO_ROOT
    file_name = os.path.basename(path)
    return os.path.join(VIDEO_ROOT, file_name)


# entry point for calculating visible height of a video file
@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json or {}
    file_path = data.get('file_path')
    rating_key = data.get('rating_key')
    plex_base_url = data.get('plex_base_url') or os.environ.get('PLEX_BASE_URL')
    plex_token = data.get('plex_token') or os.environ.get('PLEX_TOKEN')

    if not file_path and not rating_key:
        return jsonify({'error': 'file_path or rating_key required'}), 400

    if rating_key and not file_path:
        plex_path = get_file_path_from_plex(rating_key, plex_base_url, plex_token)
        if not plex_path:
            return jsonify({'error': 'could not determine file path from plex'}), 500
        # Map Plex path to container path
        file_path = map_plex_path_to_container(plex_path)

    # If given a relative path, interpret relative to VIDEO_ROOT
    if not os.path.isabs(file_path):
        file_path = os.path.join(VIDEO_ROOT, file_path)

    aspect = calculate_visible_height(file_path)
    # DEFAULT_RATIO is 1.76 in the module; keep the same comparison semantics
    valid = aspect != 1.76
    return jsonify({'aspect_ratio': aspect, 'valid': valid})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
