from flask import Flask, jsonify, request
from dotenv import load_dotenv
from flask_cors import CORS
from routes.vision_routes import vision_bp
from prometheus_flask_exporter import PrometheusMetrics
import os
import logging
import time

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

app = Flask(__name__)
CORS(app)
metrics = PrometheusMetrics(app)

# Blueprint 등록
app.register_blueprint(vision_bp, url_prefix='/vision')

# 요청 전 시간 기록
@app.before_request
def start_timer():
    request.start_time = time.time()

# 요청 후 처리 시간 로깅
@app.after_request
def log_request(response):
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        logging.info(f"{request.method} {request.path} {response.status_code} 처리시간: {duration:.3f}s")
    return response

# 글로벌 에러 핸들러
@app.errorhandler(Exception)
def handle_exception(e):
    logging.exception("Unhandled Exception: %s", e)
    response = {
        "error": "서버 내부 오류가 발생했습니다.",
        "detail": str(e)
    }
    return jsonify(response), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    logging.info(f"Starting server on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)

app = app
