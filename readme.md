# 가상 피팅 서비스

## Segmentation Model
- 논문 : https://arxiv.org/pdf/2004.12276.pdf
- 홈페이지(data set, code) : https://fashionpedia.github.io/home/Fashionpedia_download.html

### 1차 segmentation 결과
-----
1. AP 그래프
   8500 iters 부터 값이 계속 떨어진 채로 오르지 않는다. 최상의 결과가 나온 weight 부터 하이퍼 파라미터를 변경해가면서 train 해도 비슷한 결과가 나온다.
  ![그래프](https://www.notion.so/segmentation-tracking_01-520761d7902e43bcb40a09ef63cb532d?pvs=4#a91f1d1be8cf4d5f9196c7695d47906a)
  
2. AR 그래프
   AP 그래프와 마찬가지고 8500 iters 부터 값이 떨어진 채 오르지 않는다.
  ![그래프](https://www.notion.so/segmentation-tracking_01-520761d7902e43bcb40a09ef63cb532d?pvs=4#8e4c3d6eb70b43cab4a045957787353d)
  
3. segmentation 결과물
   ![이미지](https://www.notion.so/24-23-03-22-c53dbbd440b0425fb652e38cb1a12e58?pvs=4#42b9e3e41d7d4dc49f1833051acbe6b8)
  
### 2차 segmentation 결과
-----
train과 test에 들어가는 data set을 서브디렉토리를 제외한 data set으로 바꿔주었다. 
  
1. validation loss 그래프
   계속해서 하락하고 있다. 따로 overfitting이나 underfitting의 양상을 보이지 않았다.
   ![그래프](https://github.com/soy53/AIFFEL/assets/116326867/a874d416-7cce-4c68-8b68-b33d6893daab)
  
2. AP 그래프
   중간 떨어지지만 다시 올라가는 패턴을 가지고 있어 문제되지 않는다고 판단했다. 논문에 나온 baseline과도 거의 근접한 상태이다.
   ![그래프](https://github.com/soy53/AIFFEL/assets/116326867/ce4511af-c0b3-4c69-be93-9f3bcc135085)
  
3. AR 그래프
   중간에 떨어지지만 다시 올라가는 패턴이 AP 그래프보다는 확연하게 드러난다. resnet-50 FPN 6x backbone을 사용했는데 논문에는 이에 대한 AR값이 기재되어있지 않았다.
   어느 정도의 수치를 맞춰야 성능이 보장되는지 확인은 불가하지만 계속해서 학습한다면 더 좋은 결과가 나올 것이라 생각한다.
   ![그래프](https://github.com/soy53/AIFFEL/assets/116326867/eda919cc-c569-4289-b5ca-6e5bfed42699)
  
4. segmentation 결과물
   ![이미지](https://github.com/soy53/AIFFEL/assets/116326867/69423acc-0595-48ba-8ca3-ca7d26b625a3)
  
   다른 이미지들에서도 성능이 논문만큼은 아니지만 어느 정도 잘 나왔다.
   총 240000 iters (약 21 Epoch) train을 진행했는데 논문에 나온대로 90 Epoch을 train 한다면 더 좋은 결과가 나올 것이라 생각한다.
     
   그 외 결과물
   ![yn_01_segm_02](https://github.com/soy53/AIFFEL/assets/116326867/36fa90d5-e935-47d2-a449-f97c59994ebd)
   ![rw_02_segm_02](https://github.com/soy53/AIFFEL/assets/116326867/999cb01b-59da-48fc-8c7b-6eadb2734512)
   ![rw_01_segm_02](https://github.com/soy53/AIFFEL/assets/116326867/d54a5ca4-39ef-4df9-9636-107370109499)

   
