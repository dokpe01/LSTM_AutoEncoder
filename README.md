# Time Series Dafa for Anomaly Detection
<a href="https://club-project-one.vercel.app/" target="_blank">
<img src="https://www.anomalo.com/wp-content/uploads/2024/05/Machine-Learning-Approaches-to-Time-Series-Anomaly-Detection.png" alt="배너" width="100%"/>
</a>

<br/>
<br/>

# 1. Project Overview
- 프로젝트 이름: Time Series Data for Anomaly Detection
- 프로젝트 설명: 시계열 데이터의 이상치를 탐지한다.

<img src = "https://cdn.discordapp.com/attachments/1043843545632604250/1409877066433958009/tmpD12E.png?ex=68aef9d6&is=68ada856&hm=c3a84b59b2a3268eed161d9806088f1ad88f2c034605381441b31f7dd12e2152&"/>

위의 Table1은 온라인 쇼핑몰에서 사용자가 발생시킨 이벤트 로그를 간략히 나타낸 예시이다.

각 이벤트 발생 순서만을 고려한다면 로그인, 구매, 결제, 로그아웃의 일반적인 이벤트 순서를 갖는 S1 ~ S4는 정상 시퀀스이며 로그인 이벤트만이 반복적으로 일어난 S5는 이상 시퀀스이다.
그러나 원소들의 시간 간격까지 고려한다면 S1 ~ S4 역시 이상 시퀀스의 가능성을 지니게 된다.
예를 들어 S4에서 각 이벤트 사이의 시간 간격이 매우 짧다면 이는 이상 시퀀스로 판단되며, 이를 통해 매크로를 사용한 부정 구매 등을 적발할 수 있다.
이처럼 시간적 특성을 고려하여 이상 시퀀스를 탐지하는 기법을 환경시설에 설치된 각종 IoT센서에서 받아오는 시계열 데이터에 반영하고, 해당 데이터를 이용하여 미래에서 발생할 수 있는 이상치를 사전에 예측하는 것을 목표로 한다.
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

# 4. Tasks & Responsibilities
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
| HTML5    |<img src="https://github.com/user-attachments/assets/2e122e74-a28b-4ce7-aff6-382959216d31" alt="HTML5" width="100">| 
| CSS3    |   <img src="https://github.com/user-attachments/assets/c531b03d-55a3-40bf-9195-9ff8c4688f13" alt="CSS3" width="100">|
| Javascript    |  <img src="https://github.com/user-attachments/assets/4a7d7074-8c71-48b4-8652-7431477669d1" alt="Javascript" width="100"> | 
| Python 3.9    |<img src="https://i0.wp.com/junilearning.com/wp-content/uploads/2020/06/python-programming-language.webp?fit=800%2C800&ssl=1" alt="Python" width="100">| 
| PostgreSQL    |   <img src="https://dt-cdn.net/hub/logos/postgresdb-remote-monitoring.png" alt="PostgreSQL" width="100">|

<br/>
<br/>

# 6. Project Structure
```plaintext
project/
├── public/
│   ├── index.html           # UI 템플릿 파일
│   ├── model.py             # LSTM-AutoEncoder 파일
│   └── main.py              # WEB 서버 파일
├── src/
│   ├── index.css            # 전역 CSS 파일
│   └── index.js             # 엔트리 포인트 파일
├── data/
│   ├── water_pressure/      # 수압센서 데이터 파일
│   ├── turbidity/           # 수질센서 데이터 파일
│   ├── gas/                 # 유해가스센서 데이터 파일
│   ├── Vibration/           # 진동센서 데이터 파일
└───┴── flooding/            # 침수센서 데이터 파일
```

<br/>
<br/>
