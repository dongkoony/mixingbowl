from flask import Flask
from flask_cors import CORS
from app.config import Config
from app.api.routes import api_bp
from app.routes.vision_routes import vision_bp

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # CORS 설정
    CORS(app)
    
    # 블루프린트 등록
    app.register_blueprint(api_bp)
    app.register_blueprint(vision_bp, url_prefix='/api/vision')
    
    return app 