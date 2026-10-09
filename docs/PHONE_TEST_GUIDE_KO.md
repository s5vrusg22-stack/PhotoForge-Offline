# 갤럭시 S25 Ultra 실기기 테스트 안내

## 1. APK 다운로드
GitHub Actions > Build Android APK > 최신 성공한 실행 > Artifacts > PhotoForge-debug-apk 를 다운로드합니다.
ZIP을 압축 해제하고 app-debug.apk를 설치합니다.
링크: https://github.com/s5vrusg22-stack/PhotoForge-Offline/actions/workflows/android-apk.yml

**주의:** 워크플로 성공 여부와 APK 아티팩트 존재 여부를 먼저 확인해야 합니다. 저장소 소스만 있다고 APK가 생성된 것은 아닙니다.

## 2. 첫 번째 테스트 (모델 다운로드 불필요)
1. 앱 실행과 사진 열기
2. 밝기 변경, 회전, 좌우 반전
3. 투명 PNG 소품 합성 및 되돌리기
4. PNG 내보내기 후 갤러리에서 결과 확인
5. 비행기 모드에서 위 기능 반복

## 3. LaMa 모델 테스트
APK 워크플로는 LaMa ONNX 파일을 assets에 포함하려고 시도합니다.
마스크를 칠하고 물체 제거를 시도합니다. 오류가 나면 오류 메시지 전체와 화면을 캡처합니다.
사진은 원본을 따로 보관하세요.

## 4. 현재 불가능한 테스트
FLUX.2-klein LiteRT 21개 그래프 로딩, 생성형 표정·의상·자세 변경,
텍스트-이미지 생성, Galaxy S25 Ultra GPU 메모리 프로파일링은 Android 추론 엔진 미구현 상태입니다.
모델 11GB 이상을 휴대폰에 미리 받지 마세요.

## 5. 문제 보고
휴대폰 모델/Android 버전, 실행한 버튼, 예상 결과, 실제 결과, 오류 화면을 기록합니다.
크래시라면 Android Studio Logcat의 FATAL EXCEPTION 로그를 함께 제공하면 원인 파악이 빠릅니다.
