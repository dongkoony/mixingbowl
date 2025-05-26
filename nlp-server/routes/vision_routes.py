import os
import json
from flask import Blueprint, request, jsonify
from google.cloud import vision
from collections import defaultdict
import re

vision_bp = Blueprint('vision', __name__)
client = vision.ImageAnnotatorClient()

# 일반적인 키워드 필터링 (점수 감소 대상)
GENERIC_KEYWORDS = {
    'Food', 'Vegetable', 'Ingredient', 'Produce', 'Natural foods', 'Whole food',
    'Local food', 'Superfood', 'Plant', 'Vegan nutrition', 'Fruit', 'Dish', 'Cuisine',
    'Recipe', 'Meal', 'Food group', 'Staple food', 'Side dish', 'Garnish', 'Leaf vegetable',
    'Cruciferous vegetables', 'Natural material', 'Processed food', 'Raw foods', 'Organic food',
    'Health food', 'Fresh', 'Natural', 'Edible', 'Dietary', 'Nutritious', 'Healthy',
    'Vegetables', 'Vegetable', 'Fruits', 'Foods', 'Ingredients', 'food', 'vegetable',
    'ingredient', 'produce', 'product', 'item', 'material', 'stuff', 'goods', 'supply',
    'raw', 'fresh', 'natural', 'organic', 'healthy', 'nutritious', 'edible'
}

# 지역 관련 키워드 필터링 (완전 제외)
LOCATION_KEYWORDS = {
    'Ward', 'City', 'District', 'Prefecture', 'County', 'Province', 'Region',
    'Area', 'Zone', 'Territory', 'Location', 'Place', 'Spot', 'Site', 'Street',
    'Road', 'Avenue', 'Town', 'Village', 'State', 'Country', 'Market', 'Store',
    'Shop', 'Mall', 'Supermarket', 'Restaurant'
}

# 주방 도구 관련 키워드 (점수 감소 대상)
KITCHEN_TOOLS = {
    'Plate', 'Bowl', 'Cup', 'Glass', 'Utensil', 'Cookware', 'Bakeware',
    'Container', 'Tableware', 'Dishware', 'Cutlery', 'Kitchen appliance',
    'Cooking equipment', 'Storage container', 'Measuring tool', 'Dining table',
    'Table', 'Counter', 'Surface', 'Shelf', 'Cabinet', 'Drawer', 'Basket',
    'Tray', 'Cutting board', 'Knife', 'Fork', 'Spoon', 'Chopsticks'
}

# 구체적인 식재료 키워드 (가중치 증가용)
SPECIFIC_INGREDIENTS = {
    # 채소류
    '당근', 'carrot', 
    '양배추', 'cabbage',
    '토마토', 'tomato',
    '감자', 'potato',
    '양파', 'onion',
    '마늘', 'garlic',
    '상추', 'lettuce',
    '오이', 'cucumber',
    '고추', 'pepper', 'chili',
    '버섯', 'mushroom',
    '시금치', 'spinach',
    '브로콜리', 'broccoli',
    '무', 'radish',
    '배추', 'napa cabbage',
    '청경채', 'bok choy',
    '파', 'green onion', 'spring onion',
    '애호박', 'zucchini',
    '단호박', 'pumpkin',
    '고구마', 'sweet potato',
    '생강', 'ginger',
    '부추', 'chive',
    '깻잎', 'perilla leaf',
    '콩나물', 'bean sprout',
    '숙주나물', 'mung bean sprout',
    '도라지', 'bellflower root',
    '우엉', 'burdock root',
    '연근', 'lotus root',
    '방울토마토', 'cherry tomato',
    '청양고추', 'cheongyang pepper',
    '대파', 'welsh onion',
    '쪽파', 'scallion',
    
    # 과일류
    '사과', 'apple',
    '배', 'pear',
    '귤', 'tangerine',
    '오렌지', 'orange',
    '포도', 'grape',
    '딸기', 'strawberry',
    '바나나', 'banana',
    '키위', 'kiwi',
    '레몬', 'lemon',
    '망고', 'mango',
    '복숭아', 'peach',
    '자두', 'plum',
    '감', 'persimmon',
    
    # 육류
    '소고기', 'beef',
    '돼지고기', 'pork',
    '닭고기', 'chicken',
    '오리고기', 'duck',
    '양고기', 'lamb',
    '갈비', 'ribs',
    '삼겹살', 'pork belly',
    '목살', 'pork neck',
    '안심', 'tenderloin',
    '등심', 'sirloin',
    
    # 해산물
    '고등어', 'mackerel',
    '갈치', 'hairtail',
    '연어', 'salmon',
    '참치', 'tuna',
    '멸치', 'anchovy',
    '새우', 'shrimp',
    '꽃게', 'crab',
    '오징어', 'squid',
    '전복', 'abalone',
    '조개', 'clam',
    '굴', 'oyster',
    '홍합', 'mussel',
    
    # 기본 식재료
    '쌀', 'rice',
    '밀가루', 'flour',
    '계란', 'egg',
    '두부', 'tofu',
    '김치', 'kimchi',
    '된장', 'soybean paste',
    '고추장', 'red pepper paste',
    '간장', 'soy sauce',
    '참기름', 'sesame oil',
    '식용유', 'cooking oil',
    '소금', 'salt',
    '설탕', 'sugar',
    '후추', 'pepper',
    '깨', 'sesame',
    '들기름', 'perilla oil',
    '고춧가루', 'red pepper powder',
    '다진마늘', 'minced garlic',
    '다진생강', 'minced ginger'
}

def load_ko_mapping():
    try:
        with open('data/label_ko_mapping.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def should_filter_keyword(keyword):
    # 키워드 전처리
    keyword = keyword.strip().lower()
    
    # 숫자만 있는 경우 필터링
    if re.match(r'^\d+$', keyword):
        return True
        
    # 너무 짧은 키워드 필터링 (3글자 미만)
    if len(keyword) < 3:
        return True
        
    # 특수문자만 있는 경우 필터링
    if re.match(r'^[^a-zA-Z0-9가-힣]+$', keyword):
        return True
    
    # 지역 관련 키워드는 완전히 제외
    if keyword in LOCATION_KEYWORDS or any(loc.lower() in keyword for loc in LOCATION_KEYWORDS):
        return True
        
    return False

def adjust_score(keyword, base_score):
    """키워드에 따라 점수를 조정하는 함수"""
    keyword = keyword.lower()
    
    # 구체적인 식재료는 점수 증가
    if keyword in SPECIFIC_INGREDIENTS or keyword in [x.lower() for x in SPECIFIC_INGREDIENTS]:
        return base_score * 1.8  # 가중치 증가
        
    # 일반적인 키워드는 점수 감소
    if keyword in GENERIC_KEYWORDS:
        return base_score * 0.2  # 더 많이 감소
        
    # 주방 도구는 점수 감소
    if keyword in KITCHEN_TOOLS:
        return base_score * 0.3  # 더 많이 감소
        
    return base_score

def get_weighted_keywords(image_file):
    with open(image_file, 'rb') as image:
        content = image.read()

    image = vision.Image(content=content)
    
    # 이미지 분석 설정
    features = [
        vision.Feature(type_=vision.Feature.Type.WEB_DETECTION, max_results=50),
        vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=50),
        vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=50),
        vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
        vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES)
    ]
    
    # 통합 이미지 분석 요청
    response = client.annotate_image({'image': image, 'features': features})
    
    keywords = defaultdict(float)
    
    # 웹 감지 결과 (가중치: 1.5)
    if response.web_detection:
        for entity in response.web_detection.web_entities:
            if entity.score >= 0.3:  # 임계값 더 낮춤
                keyword = entity.description.lower()
                if not should_filter_keyword(keyword):
                    base_score = entity.score * 1.5
                    keywords[keyword] += adjust_score(keyword, base_score)

    # 객체 감지 결과 (가중치: 1.3)
    for obj in response.localized_object_annotations:
        if obj.score >= 0.3:  # 임계값 더 낮춤
            keyword = obj.name.lower()
            if not should_filter_keyword(keyword):
                base_score = obj.score * 1.3
                keywords[keyword] += adjust_score(keyword, base_score)

    # 레이블 감지 결과 (가중치: 1.2)
    for label in response.label_annotations:
        if label.score >= 0.3:  # 임계값 더 낮춤
            keyword = label.description.lower()
            if not should_filter_keyword(keyword):
                base_score = label.score * 1.2
                keywords[keyword] += adjust_score(keyword, base_score)

    # OCR 텍스트 분석 (가중치: 1.0)
    if response.text_annotations:
        text_blocks = response.text_annotations[0].description.split('\n')
        for block in text_blocks:
            words = block.split()
            for word in words:
                keyword = word.lower()
                if len(keyword) >= 3 and not should_filter_keyword(keyword):
                    base_score = 1.0
                    keywords[keyword] += adjust_score(keyword, base_score)

    # 유사/중복 키워드 통합 및 정규화
    final_keywords = defaultdict(float)
    processed_keywords = set()
    
    # 정규화를 위한 최대 점수 계산
    max_score = max(keywords.values()) if keywords else 1.0
    
    for keyword, score in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
        if keyword in processed_keywords:
            continue
            
        # 유사한 키워드 찾기
        similar_keywords = {keyword}
        for other_keyword in keywords:
            if other_keyword != keyword and not should_filter_keyword(other_keyword):
                # 완전히 포함된 경우나 복합 키워드인 경우
                if (keyword == other_keyword or 
                    keyword in other_keyword.split() or 
                    other_keyword in keyword.split()):
                    similar_keywords.add(other_keyword)
                    
        # 구체적인 식재료 키워드 우선 선택
        best_keyword = max(similar_keywords, 
                         key=lambda k: (k in SPECIFIC_INGREDIENTS, len(k.split()), keywords[k]))
        
        # 점수 정규화 (0-1 범위로)
        normalized_score = keywords[best_keyword] / max_score
        
        # 신뢰도 임계값 낮춤
        if normalized_score >= 0.3:  # 더 낮은 임계값
            final_keywords[best_keyword] = normalized_score
            processed_keywords.update(similar_keywords)

    # 결과 생성
    ko_mapping = load_ko_mapping()
    result = []
    
    for keyword, score in sorted(final_keywords.items(), key=lambda x: x[1], reverse=True):
        # 일반적인 키워드는 건너뛰기
        if keyword.lower() in GENERIC_KEYWORDS and score < 0.5:
            continue
            
        entry = {
            'keyword': keyword,
            'score': round(score * 100, 1)  # 백분율로 변환
        }
        if ko_mapping and keyword.lower() in ko_mapping:
            entry['korean'] = ko_mapping[keyword.lower()]
        result.append(entry)

    return result[:15]  # 상위 결과 수를 15개로 유지

@vision_bp.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if not image_file.filename:
        return jsonify({'error': 'No selected file'}), 400

    # 허용된 파일 확장자 검사
    allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}
    if '.' not in image_file.filename or \
       image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # 임시 파일로 저장
        temp_path = f"/tmp/{image_file.filename}"
        image_file.save(temp_path)
        
        # 이미지 분석
        keywords = get_weighted_keywords(temp_path)
        
        # 임시 파일 삭제
        os.remove(temp_path)
        
        return jsonify({
            'status': 'success',
            'keywords': keywords
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 