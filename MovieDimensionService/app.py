# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import os
from movie_dimension import calculate_visible_height, get_file_path_from_plex

app = Flask(__name__)

# In Synology the shared folder is mounted at /video inside the container 
VIDEO_ROOT = os.environ.get('VIDEO_ROOT', '/video')

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
        # Map Plex path to container path: assume Plex used /volumeX/video and container mounts that to /video
        # If plex provides absolute path, we try to take the filename and use VIDEO_ROOT
        file_name = os.path.basename(plex_path)
        file_path = os.path.join(VIDEO_ROOT, file_name)

    # If given a relative path, interpret relative to VIDEO_ROOT
    if not os.path.isabs(file_path):
        file_path = os.path.join(VIDEO_ROOT, file_path)

    aspect = calculate_visible_height(file_path)
    # DEFAULT_RATIO is 1.76 in the module; keep the same comparison semantics
    valid = aspect != 1.76
    return jsonify({'aspect_ratio': aspect, 'valid': valid})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
