import requests
import os
from pathlib import Path

def test_vision_api():
    """
    Vision API 테스트 함수
    """
    # Docker 환경의 API 엔드포인트 설정
    url = 'http://localhost:5001/api/vision/analyze-image'  # 새로운 엔드포인트 사용
    
    # 프로젝트 루트 디렉토리 기준으로 이미지 경로 설정
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    image_path = project_root / 'data' / 'images' / 'image1.jpg'
    
    if not image_path.exists():
        print(f"경고: 이미지 파일을 찾을 수 없습니다.")
        print(f"확인할 경로: {image_path}")
        return
            
    print(f"\n이미지 분석 테스트 시작: {image_path.name}")
    print(f"이미지 경로: {image_path}")
    print(f"요청 URL: {url}")
    
    try:
        # 이미지 파일 업로드
        with open(image_path, 'rb') as image_file:
            # Vision API는 'image'라는 이름으로 파일을 받도록 설정되어 있음
            files = {'image': (image_path.name, image_file, 'image/jpeg')}
            print("\nAPI 요청 전송 중...")
            response = requests.post(url, files=files)
        
        # 응답 확인
        if response.status_code == 200:
            result = response.json()
            print("\n✅ 감지된 레이블:")
            for label in result.get('labels', []):
                if 'description' in label:
                    print(f"- {label['description']} (신뢰도: {label['score']:.2%})")
                    print(f"  중요도: {label['topicality']:.2%}")
            
            if 'objects' in result:
                print("\n✅ 감지된 객체:")
                for obj in result['objects']:
                    print(f"- {obj['name']} (신뢰도: {obj['confidence']:.2%})")
                    if 'bounding_box' in obj:
                        print(f"  위치: 좌상단({obj['bounding_box']['left']:.2f}, {obj['bounding_box']['top']:.2f}), "
                              f"우하단({obj['bounding_box']['right']:.2f}, {obj['bounding_box']['bottom']:.2f})")
        else:
            print(f"\n❌ API 오류 발생 (상태 코드: {response.status_code})")
            print("오류 내용:")
            print(response.json())
            print("\n디버깅 정보:")
            print(f"요청 URL: {url}")
            print(f"요청 파일: {image_path.name}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ 서버 연결 실패")
        print("1. nlp-server가 실행 중인지 확인하세요")
        print("2. Docker 컨테이너가 정상적으로 실행 중인지 확인하세요")
        print("3. 포트 5001이 접근 가능한지 확인하세요")
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {str(e)}")

if __name__ == '__main__':
    print("=== Vision API 테스트 ===")
    print("\n실행 전 확인사항:")
    print("1. Docker 컨테이너 (mixingbowl-nlp_server)가 실행 중인지")
    print("2. Google Cloud 인증이 정상적으로 설정되었는지")
    print("3. 테스트 이미지 파일이 존재하는지")
    
    input("\nEnter를 눌러 테스트를 시작하세요...")
    test_vision_api() 