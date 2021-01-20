What i did
======================

# 1. 법령 정제

[국가법령정보센터](https://www.law.go.kr/LSW//main.html)에서 얻은 법의 내용 쳇봇의 답변으로 사용하기 위하여 .xlsx 파일 형식으로 깔끔하게 정리. 예시 파일로 [고압가스 안전관리법](https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EA%B3%A0%EC%95%95%EA%B0%80%EC%8A%A4%EC%95%88%EC%A0%84%EA%B4%80%EB%A6%AC%EB%B2%95)을 사용한다.

# 2. 의료 영상 전처리 

MRI 영상에 대해 몇 가지 정규화 방식을 적용하는 코드 

참고 링크
* https://github.com/jcreinhold/intensity-normalization

# 3. Variational AutoEncoder (VAE) 

비지도 이상치 탐색 (Unsupervised Anomaly Detection) 방법중의 하나인 Variational AutoEncoder에 대한 코드

참고 링크
* https://keraskorea.github.io/posts/2018-10-23-keras_autoencoder/
* https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/