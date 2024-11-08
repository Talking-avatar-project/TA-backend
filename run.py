from app import create_app
from app.controllers.facial_recognition_controller import facial_recognition_bp
from app.controllers.chatbot_controller import chatbot_bp
from app.controllers.avatar_controller import avatar_bp

app = create_app()

# Registrar Blueprints
app.register_blueprint(facial_recognition_bp, url_prefix='/facial_recognition')
app.register_blueprint(chatbot_bp, url_prefix='/chatbot')
app.register_blueprint(avatar_bp, url_prefix='/avatar')

if __name__ == '__main__':
    app.run(debug=True)
