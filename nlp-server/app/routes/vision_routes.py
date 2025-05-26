from flask import Blueprint, request, jsonify
from google.cloud import vision
import os
from dotenv import load_dotenv
import logging
import json
from typing import List, Dict, Any

# 로깅 설정
logger = logging.getLogger(__name__)

load_dotenv()

vision_bp = Blueprint('vision', __name__)

# 일반적인 카테고리 키워드 (필터링에 사용)
GENERIC_KEYWORDS = {
    'food', 'ingredient', 'produce', 'vegetable', 'fruit', 'dish', 'cuisine',
    'meal', 'snack', 'natural foods', 'food group', 'tableware', 'kitchen'
}

try:
    client = vision.ImageAnnotatorClient()
    # 한글 레이블 매핑 로드
    with open("data/label_ko_mapping.json", "r", encoding="utf-8") as f:
        LABEL_KO_MAPPING = json.load(f)
except Exception as e:
    logger.error(f"초기화 실패: {str(e)}")
    client = None
    LABEL_KO_MAPPING = {}

def get_korean_label(eng_label: str) -> str:
    """영문 레이블에 대응하는 한글 레이블을 반환합니다."""
    return LABEL_KO_MAPPING.get(eng_label.lower(), eng_label)

def filter_and_score_keywords(labels: List[Dict[str, Any]], 
                            objects: List[Dict[str, Any]], 
                            web_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """키워드를 필터링하고 점수를 계산합니다."""
    
    # 모든 키워드를 하나의 리스트로 통합
    all_keywords = []
    
    # 웹 엔티티 처리 (가중치 1.5 적용)
    for entity in web_entities:
        if entity['description_en'].lower() not in GENERIC_KEYWORDS:
            all_keywords.append({
                'keyword_en': entity['description_en'],
                'keyword_ko': entity['description_ko'],
                'score': float(entity['score']) * 1.5,
                'source': 'web'
            })
    
    # 객체 감지 결과 처리 (가중치 1.2 적용)
    for obj in objects:
        if obj['name_en'].lower() not in GENERIC_KEYWORDS:
            all_keywords.append({
                'keyword_en': obj['name_en'],
                'keyword_ko': obj['name_ko'],
                'score': float(obj['confidence']) * 1.2,
                'source': 'object'
            })
    
    # 레이블 처리 (가중치 1.0)
    for label in labels:
        if label['description_en'].lower() not in GENERIC_KEYWORDS:
            all_keywords.append({
                'keyword_en': label['description_en'],
                'keyword_ko': label['description_ko'],
                'score': float(label['score']),
                'source': 'label'
            })
    
    # 중복 제거 (가장 높은 점수 유지)
    unique_keywords = {}
    for kw in all_keywords:
        key = kw['keyword_en'].lower()
        if key not in unique_keywords or unique_keywords[key]['score'] < kw['score']:
            unique_keywords[key] = kw
    
    # 점수순으로 정렬
    sorted_keywords = sorted(unique_keywords.values(), key=lambda x: x['score'], reverse=True)
    
    return {
        'all_keywords': sorted_keywords,
        'top_keywords': sorted_keywords[:5]  # 상위 5개만 선택
    }

@vision_bp.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        if client is None:
            return jsonify({'error': 'Vision API 클라이언트가 초기화되지 않았습니다.'}), 500

        if 'image' not in request.files:
            return jsonify({'error': '이미지 파일이 필요합니다.'}), 400

        image_file = request.files['image']
        
        # 파일 확장자 검증
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if '.' not in image_file.filename or \
            image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400

        content = image_file.read()
        image = vision.Image(content=content)
        
        # Vision API 호출
        label_response = client.label_detection(image=image, max_results=20)
        object_response = client.object_localization(image=image)
        web_response = client.web_detection(image=image)
        
        # 결과 변환
        labels = [
            {
                'description_en': label.description,
                'description_ko': get_korean_label(label.description),
                'score': float(label.score),
                'topicality': float(label.topicality)
            }
            for label in label_response.label_annotations
        ]
        
        objects = [
            {
                'name_en': obj.name,
                'name_ko': get_korean_label(obj.name),
                'confidence': float(obj.score),
                'bounding_box': {
                    'left': obj.bounding_poly.normalized_vertices[0].x,
                    'top': obj.bounding_poly.normalized_vertices[0].y,
                    'right': obj.bounding_poly.normalized_vertices[2].x,
                    'bottom': obj.bounding_poly.normalized_vertices[2].y
                }
            }
            for obj in object_response.localized_object_annotations
        ]
        
        web_entities = [
            {
                'description_en': entity.description,
                'description_ko': get_korean_label(entity.description),
                'score': float(entity.score) if entity.score else 0.0
            }
            for entity in web_response.web_detection.web_entities
            if entity.description
        ]
        
        # 키워드 필터링 및 점수 계산
        keywords_result = filter_and_score_keywords(labels, objects, web_entities)
        
        # 최종 결과 구성
        results = {
            'keywords': keywords_result['top_keywords'],  # 상위 5개 키워드
            'all_keywords': keywords_result['all_keywords'],  # 전체 키워드
            'raw_data': {  # 원본 데이터
                'labels': labels,
                'objects': objects,
                'web_entities': web_entities
            }
        }

        logger.info(f"이미지 분석 완료: {len(keywords_result['all_keywords'])} 개의 키워드 감지됨")
        return jsonify(results)

    except Exception as e:
        logger.error(f"이미지 분석 중 오류 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500