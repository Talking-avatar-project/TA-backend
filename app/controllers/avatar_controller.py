# controllers/avatar_controller.py
from flask import Blueprint, request, jsonify
from app.services.avatar_service import AvatarService

avatar_bp = Blueprint('avatar', __name__)


@avatar_bp.route('/avatar/express', methods=['POST'])
def express():
    data = request.get_json()
    emotion_data = data.get('emotion_data')
    if not emotion_data:
        return jsonify({"error": "No emotion data provided"}), 400

    response = AvatarService.generate_expression(emotion_data)
    return jsonify(response)
