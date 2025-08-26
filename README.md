# Time Series Dafa for Anomaly Detection
<a href="https://club-project-one.vercel.app/" target="_blank">
<img src="https://www.anomalo.com/wp-content/uploads/2024/05/Machine-Learning-Approaches-to-Time-Series-Anomaly-Detection.png" alt="배너" width="100%"/>
</a>

<br/>
<br/>

# 1. Project Overview
- 프로젝트 이름: 시계열 데이터 이상치 탐지
- 프로젝트 설명: 환경시설에 설치된 각종 센서에서 시계열 데이터를 수집하고, 해당 데이터를 이용하여 미래에서 발생할 수 있는 이상치를 사전에 예측한다.

<br/>
<br/>

# 2. Members
| 허성욱 | 박미영 | 이하린 |
|:------:|:------:|:------:|
| <img src="https://avatars.githubusercontent.com/u/225093840?v=4" alt="허성욱" width="150">  | <img src="https://postfiles.pstatic.net/MjAyMTA0MjlfMTEy/MDAxNjE5Njg1NjQ0OTMw.yiop5mDbIRN6e9ieuLdz-FU5wykhpZbftuw_fq6fQlUg.PLBDXChK7ID_Ypnm95_X987Pqwz35HwpZiRc8ijBbWEg.PNG.hanyang7117/2.png?type=w966" alt="박미영" width="150"> |  <img src="https://postfiles.pstatic.net/MjAyNDEwMDhfMTE1/MDAxNzI4Mzc4NDA3NDcw.G78Vpz7np1TO8_z6Ox7zfOOvMo9Dmzvjld9olnymdvQg.gTtPw6EM-auC8sy6liJFqAUKCwmZuAXvz4uwqUJogIcg.JPEG/common.jpeg?type=w966" alt="이하린" width="150"> | 
| 경남대학교 | 경남대학교 | 부산대학교 |
| [GitHub](https://github.com/dokpe01) | None | None | 

<br/>
<br/>

# 3. Key Features
- **Data Preprocessing**:
  - 정확한 분석 결과 도출과 모델의 성능을 향상시키기 위해 데이터 전처리 과정을 거친다.
  - 결측치 처리, 이상치 제거, 데이터 정규화 등
    
<img src="https://velog.velcdn.com/images/hyeda/post/cef78295-1a18-43c9-9cb6-90a89a7673cf/image.png" alt="autoencoder" width="70%"/>

- **Stacked AutoEncoder**:
  - 입력된 데이터를 재복원하는 비지도학습으로, 입력된 값과 최대한 비슷하게 출력되도록 재구성한다.
  - AutoEncoder는 정상 데이터만을 사용하여 학습하고, 비정상 데이터는 학습되지 않았기에 입력 데이터에 비정상 데이터가 입력된다면 복원된 데이터가 입력 데이터와 많은 차이를 보이게 된다. 해당 차이를 복원 에러(Reconstruction Error)라고 정의한다.
  - 임계치(Threshold)를 정하고, 복원 에러가 임계치를 넘어서게 되면 이상치 탐지를 하게 된다. (Anomaly Detection)

<img src="https://anencore94.github.io/assets/images/2020-10-13/seq2seq.png" alt="lstm" width="70%"/>

- **LSTM (Long Short Term Memory)**:
  - LSTM은 순환신경망(Recurrent Neural Network, RNN)의 한 종류로서, 과거에 발생한 이벤트가 이후 이벤트 발생에 영향을 주는 구조로 이루어져 있다. 따라서 이는 선후 관계가 존재하는 시퀀스 데이터를 다루기에 매우 적합하다.
  - 기존의 순환신경망은 데이터의 크기가 커질수록 비교적 먼 관계에 입력된 이벤트에 대한 학습 결과가 현재 이벤트 예측에 제대로 반영되지 않는 장기 의존성(Long-Term Dependency) 문제를 가지고 있었다. LSTM은 이를 기존 순환신경망 셀(cell)에 구조적인 변화를 주어 해결하였다.

- **Curve Shifting**:
  - 단순히 현재 시점의 error를 계산하여 비정상 신호를 탐지하는 것은 이미 고장이 발생한 후 예측하는 것과 다름이 없기 때문에 데이터에 대한 시점 변환이 꼭 필요하다.
  - 시계열 데이터의 시간적 특성을 활용하여 사전 예측 개념을 적용할 수 있다. 이는 미래에 IoT 센서에서 받아올 데이터에서 이상치가 발생할 조짐을 예측하는 것과 같은 개념이다.
  - Curve Shifting을 통해 데이터의 시점을 변환해주고 normal 데이터만을 통해 LSTM-AutoEncoder 모델을 학습시킨다. 그 후 Reconstruction Error를 계산 후 Precision Recall Curve를 통해 normal/abnormal을 구분하기 위한 Threshold를 지정하게되고 이 Threshold를 기준으로 마지막 테스트셋의 Reconstruction Error를 분류하여 t+n 시점을 예측하게 된다.

<br/>
<br/>

# 4. Tasks & Responsibilities (작업 및 역할 분담)
|  |  |  |
|-----------------|-----------------|-----------------|
| 허성욱    |  <img src="https://avatars.githubusercontent.com/u/225093840?v=4" alt="허성욱" width="100"> | <ul><li>프로그래밍</li></ul>     |
| 박미영   |  <img src="https://postfiles.pstatic.net/MjAyMTA0MjlfMTEy/MDAxNjE5Njg1NjQ0OTMw.yiop5mDbIRN6e9ieuLdz-FU5wykhpZbftuw_fq6fQlUg.PLBDXChK7ID_Ypnm95_X987Pqwz35HwpZiRc8ijBbWEg.PNG.hanyang7117/2.png?type=w966" alt="박미영" width="100">| <ul><li>리딩 및 커뮤니케이션</li><li>프로젝트 계획 및 관리</li></ul> |
| 이하린   |  <img src="https://postfiles.pstatic.net/MjAyNDEwMDhfMTE1/MDAxNzI4Mzc4NDA3NDcw.G78Vpz7np1TO8_z6Ox7zfOOvMo9Dmzvjld9olnymdvQg.gTtPw6EM-auC8sy6liJFqAUKCwmZuAXvz4uwqUJogIcg.JPEG/common.jpeg?type=w966" alt="이하린" width="100">    |<ul><li>IoT센서 데이터 제공</li></ul>  |

<br/>
<br/>

# 5. Technology Stack
|  |  |
|-----------------|-----------------|
| Python    |<img src="https://pythonlife.in/images/pythonlogo.png" alt="Python" width="100">| 
| PostgreSQL    |   <img src="https://dt-cdn.net/hub/logos/postgresdb-remote-monitoring.png" alt="PostgreSQL" width="100">|

<br/>
<br/>

# 6. Project Structure
```plaintext
project/
├── public/
│   ├── index.html           # HTML 템플릿 파일
│   ├── lstm.py              # LSTM-AutoEncoder 파일
│   └── main.py              # Flask 서버 파일
├── data/
│   ├── water_pressure/      # 수압센서 데이터 파일
│   ├── turbidity/           # 수질센서 데이터 파일
│   ├── gas/                 # 유해가스센서 데이터 파일
│   ├── Vibration/           # 진동센서 데이터 파일
└───┴── flooding/            # 침수센서 데이터 파일
```

<br/>
<br/>
