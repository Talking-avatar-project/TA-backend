# services/chatbot_service.py
from app.utils.dialogflow_utils import detect_intent_text

class ChatbotService:
    @staticmethod
    def get_response(user_message):
        # Llamada a Dialogflow o cualquier otro servicio de NLP configurado
        return detect_intent_text(user_message)
