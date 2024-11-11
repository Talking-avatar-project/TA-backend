# controllers/chatbot_controller.py
from flask import Blueprint, request, jsonify
from app.services.chatbot_service import (ChatbotService)

chatbot_bp = Blueprint('chatbot', __name__)


@chatbot_bp.route('/chatbot/respond', methods=['POST'])
def respond():
    data = request.get_json()
    user_message = data.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    response = ChatbotService.get_response(user_message)
    return jsonify({"response": response})
