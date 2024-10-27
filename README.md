# 모델 (server 디렉토리)

### 요구사항
- `python`(3.11.9)

### 실행 방법

1. 필수 라이브러리 설치
```shell
cd server/scripts
pip install -r requirements.txt
```

2. XGB 모델 훈련 진행 (.py, .ipynb 모두 지원) </br>
```shell
# config_train에서 훈련 관련 세부 사항 수정 가능
cd server/src
python train.py # 또는 notebooks/train.ipynb 실행
```

3. 시뮬레이션 시각화 진행 (Optional)
```
# config_simulation에서 시뮬레이션 시각화 관련 세부 사항 수정 가능
python simulation -c ../scripts/config_simulation.ini [-d <distance_rate>] [-dr <demand_rate>] [-cr <competition_rate>] -m simulation
```
  Arguments
  - `-d <distance_rate>` 거리에 부여할 가중치 (required, 0~1 사이의 float)
  - `-dr <demand_rate>` 예상 수요량에 부여할 가중치 (required, 0~1 사이의 float)
  - `-cr <competition_rate>` 클러스터 주변 경쟁률에 부여할 가중치 (required, 0~1 사이의 float)

4. 가중치 최적화 진행
```
python simulation [-c <config_file>] -m optimization [-n <naive>] 
```
  Arguments
  - `-c <config_file>` 최적화 하고 싶은 상황 (required, ../scripts/config_holidays.ini, ../scripts/config_weekdays.ini, ../scripts/config_weekends.ini 중 선택)
  - `-n <naive>` 배차 알고리즘 없이 실험하고 싶은지 여부 (optional, True시 배차 알고리즘 없이 1회 진행)
  

# 대시보드 (dashboard 디렉토리)

### 요구사항

- `python`(3.10 이상 권장)
- `node.js`

### 실행 방법

1. 모델 `src` 디렉토리의 서버 파일 `taxi_server.py` 실행 (기본 포트 `5980`)
2. `dashboard/src/socket.js`의 `URL` 변수를 앞서 실행한 서버 파일이 구동되는 IP 주소와 포트 번호로 수정
3. 아래 코드 실행

```shell
npm install
npm start
```

# 앱 클라이언트 (client 디렉토리)

### 요구사항 (Android 기준)

- Flutter 구동이 가능한 `Android` 버전
- Flutter SDK (3.22 이상 권장)
- Android SDK

### 설치 방법

- `prebuilt.apk` 설치

### 빌드 방법 (위 방법대로 설치 시 불필요)

1. `client` 디렉토리에서 아래 코드 실행

```shell
flutter pub get
dart run build_runner build -d
flutter build apk
adb install build/app/outputs/flutter-apk/app-release.apk # 혹은 해당 apk 파일을 직접 휴대폰에서 설치
```

### 실행 방법

1. 앱이 실행되면 위치 권한 부여 및 소켓 URL 주소를 앞서 실행한 서버 파일이 구동되는 IP 주소와 포트 번호로 수정
2. `Client` 버튼을 눌러 실행
