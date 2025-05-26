# test_vision.py
import requests

def test_vision_api():
    url = 'http://localhost:5001/api/search/image'
    
    # 이미지 파일 경로
    image_path = 'nlp-server/data/images/image1.jpg'
    
    # 파일 업로드
    with open(image_path, 'rb') as image_file:
        files = {'file': image_file}
        response = requests.post(url, files=files)
    
    # 결과 출력
    print(response.json())

if __name__ == '__main__':
    test_vision_api()