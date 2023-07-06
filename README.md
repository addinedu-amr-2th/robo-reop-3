# 프로젝트 소개
----------------------
<p align="center">
  <img src="https://github.com/addinedu-amr-2th/robo-reop-3/assets/117617384/a47b33db-8e67-4869-9070-062e9dd757f5">
</p>
로봇의 카메라를 와 OpenCV를 이용하여 모터를 제어하는 프로젝트

# 시연 영상
---------------------------
<p align="center">
  <img src="https://github.com/addinedu-amr-2th/robo-reop-3/assets/117617384/1d556166-901f-4862-bec8-a2010ec72a7f">
</p>

# 목적
-------------------------
-실제 도로와 비슷한 주행로 환경을 구현하여 로봇의 이미지 만을 가지고 차선의 중앙을 찾아내서 주행해보자



-딥 러닝 모델을 사용하지 않은 상태로 차선을 인식하여 주행

# 프로젝트 설명
--------------
<p align="center">
  <img src="https://github.com/addinedu-amr-2th/robo-reop-3/assets/117617384/f721a075-a70f-4dd3-b18c-a030ef72f12b">
</p>

*전체 시스템 구성도<br/>
  *(우) 실제 로봇의 전원을 키고 실행 시켜을때를 간단하게 나타낸 것입니다.


  
  *(좌) 카메라의 정보를 읽어 들여서 여러가지 필더를 이용하여 아래 2번 사진의 두개의 선의 픽셀 차이값을 가지고 로봇의 모터를 제어한다.
  <br/>  만약 1번 사진 처럼 왼쪽 차선과 오른쪽 차선의 인식양을 인식하다가 표준값 보다 픽셀양이 적다면 한쪽으로 치우쳐져 있다고 판단하여 모터를 제어한다.
  <br/>  3번 사진은 교차로를 만나을때의 사진인데 사진 상단의 양쪽 픽셀 값이 없어진다면 직진을 하도록 제어 하였습니다.
  
<p align="center">
  <img src="https://github.com/addinedu-amr-2th/robo-reop-3/assets/117617384/882cfd3d-dabb-4db8-aaa5-4d060b2162e2">
</p>


# dasd
-dad
  -das
