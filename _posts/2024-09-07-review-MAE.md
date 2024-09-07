---
title: Masked Autoencoders are Scalable Vision Learners Review
layout: post
---

<img src="https://github.com/user-attachments/assets/fe89a996-b849-41d2-9bca-94472745e4f6" alt="MAE figure" class="post-pic"/>
<br />
<br />

[Masked Autoencoders are Scalable Vision Learners 논문](https://arxiv.org/abs/2111.06377)(2021)

## Introduction

- 하드웨어의 발전으로 어마어마한 용량의 학습이 가능해짐. 그러나 데이터는 이를 따라오는가? 라벨이 부족
- NLP에서는 이걸 이미 해결함.GPT(auto-regressive), BERT(masked autoencoding): 둘 다 데이터 일부를 가리고 학습함. 100B 이상의 파라미터 학습으로 가는 중
- 그럼 비전에서도 할 수 있는거 아니야? 왜 안함?
    - 최근까지 두 분야가 쓰는 모델이 달랐음. 하지만 트랜스포머 때매 경계가 무너졌다.
    - 정보 밀도의 차이: 언어란 인간이 만들었으므로 정보가 매우 빽빽함. 이미지는 자연에서 나왔으므로 밀도가 부족함. 그래서 걍 학습하면 주변 정보를 쉽게 메꿔버림
    - 오토인코더의 **디코더**는 언어와 이미지 분야에서 다른 역할을 함: 언어모델은 **단어**를 복구하는 반면 이미지모델은 **픽셀**을 복구함. 복구하는 정보량의 차원이 다름;
- 정보 밀도의 차이를 어떻게 극복하는가? 이미지를 **엄청 많이 비우면** 됨. 이럼 모델에게 상당히 어려운 문제가 되어 자가지도 학습이 가능함
- 디코더 모델의 차이는? 디코더 디자인을 **잘** 해야함
- 여기까지 읽다가 느낀 점: 이 사람들 생각을 꽤 깊게 하는구나. 자연스러운 사고지만 사고를 했기 때문에 결론을 얻을 수 있는 것. 콜럼버스의 달걀
- 느낀 점 2: gemini 등이 해주는 논문 요약이 실제로 내용을 완전히 이해하기에 충분한가? 75%라는 단순한 수치가 중요하단 생각은 안 했었는데 본문을 읽다보니 중요하단 점을 알았음. 아니면 내가 gemini의 이 단순한 요약을 제대로 안 읽었던 걸까. 요약을 읽는 것만으론 사람은 글의 요지를 이해할 수 없는 건가?
    - This is a paper about a method for self-supervised learning of computer vision models. It discusses the use of masked autoencoders (MAE). The authors propose a simple approach where random patches of an input image are masked and then the missing pixels are reconstructed. They use an asymmetric encoder-decoder architecture and find that masking a large proportion of the input image works well. They report that their method is efficient and effective, allowing them to train large models that achieve good results on downstream tasks. 라는 요약을 봤을 때 masking large proportion이 핵심 포인트가 아닌 것처럼 이해할 수도 있잖음…
- 이런 마스킹 방식을 쓰면 학습량을 엄청 줄일 수 있으면서도 효율적으로 학습할 수 있음(인풋을 줄였기 때문에..) 사전 학습 시간과 사용 메모리도 3배 이상 줄어듦
- 데이터를 많이 필요로 하는 모델에도 적용할 수 있으며, 일반화 성능도 좋아지고, transfer learning도 잘 됨. 이건 사실 NLP에서도 이랬기 때문에 잘 될 만 하다

## 관련 연구

- 마스크드 언어모델: GPT, BERT
- 오토인코더: 클래식임. PCA랑 k-means도 오토인코더임(?!)
- 마스크드 이미지 인코더
- 자기지도 학습: 많이 핫하지만 최근엔 특히 Contrastive learning이 유사도를 활용하는 게 제일 핫했음. 다만 얘네는 augmentation이 많이 필요함

## 접근

- 읽기 전에 생각한거: 다른 건 알겠는데 디코더 디자인? 이건 좀 집중해서 봐야할 듯
- 인코더 디코더 디자인은 클래식 오토인코더랑 동일하지만, 인코더는 안짤린 패치, 디코더는 작업된 latent rep.+마스크 토큰 2개에 작용하는 **비대칭적인 디자인**을 채용했다.
- 마스킹: uniform하게 마스크 씌웠어. 마스크 비율도 너무 쉽게 근처에서 추론 못하게 높게 잡았어.
- 인코더: ViT
- 디코더: 인코딩된 보이는 패치 토큰+마스크토큰에 작용. 마스크 토큰은 공유되고 학습가능. 각 토큰 전체에 positional embedding도 더함(위치는 알아야제..) 디코더도 트랜스포머
- 디코더는 복원만 학습함. 인코더랑 관련이 없어도 됨. 인코더보다 계산량이 훨씬 적어도 됨. 그래서 사전학습시간을 크게 줄일 수 있음
- Loss는 MSE인데 BERT처럼 마스크 패치에만 적용함. 이건 좀 특이한 듯?
- 픽셀을 normalize하면 좀 더 좋더라
- 단순한 실행법: 이미지 패치를 토큰으로 바꾸고, 토큰을 섞고, 섞은 토큰들 중 뒷줄 제거.
    
    인코딩 이후엔 뒷줄에 마스크 토큰을 넣고, 역섞기로 복원해서 디코더에 넣음.
    

## 이미지넷 실험

- 이미지넷-1k로 자기지도로 사전학습을 수행한 후, 평가하기 위해 지도학습을 함. 이걸로 linear probing이랑 end-to-end fine-tuning을 함.(Linear probing이 뭐야?)
- **Linear probing**: backbone model 얼리고 FCN 1개만 달아서 학습하는 것↔모든 백본을 학습하는 것이 **fine-tuning**
- 마스킹 비율: fine tuning은 좀 둔감하긴 했는데, 쨌든 둘다 75% 가리면 엄청 좋았다.
- decoder depth: linear probing에서는 depth가 깊으면 좋긴 했지만, 전반적으로 depth 사양을 잘 안탔음. 심지어 fine-tuning에선 1블록 디코더마저 먹힘
- decoder width: 얇아도 fine-tuning은 잘됨
- 결론: 디코더가 작아도 됨. 모든 토큰에 작용하지만, 인코더에 비해서 계산량이 매우 적음
- 복원 타겟을 다양하게 잡을 수 있지만, 픽셀이 젤 낫더라
- augmentation: 없어도 됨;; 다른 연구(costrastive learning)이랑 대조적임. 왜인지도 직관적인게, 마스킹 자체가 augmentation 효과를 냄
- 마스킹: 걍 유니폼 랜덤 샘플링이 젤 좋았음
- 에폭수: 걍 많을수록 계속 좋아짐 수렴도 안함; 그래서 사전학습은 800에폭 이상 했음

### 이전 연구와 비교

### 부분 fine-tuning

마지막 블록만 학습해도 확 좋아짐; linear probing은 너무 뒤쳐졌다. 절반만 fine-tuning해도 좋음

## Transfer learning 실험

우리가 다 좋았음

- **실험 디테일은 나중에 코드로 직접 확인**하면서 체크해보는 것도 좋을 듯
- [https://velog.io/@xuio/Transfer-Learning과-Fine-Tuning](https://velog.io/@xuio/Transfer-Learning%EA%B3%BC-Fine-Tuning)
논문 보면서 헷갈렸는데 보고 다시 정리해두자

## 결론

- 이미지와 언어는 다른 형태의 시그널이므로, 다룰 때 다른 방식을 고민하는 것이 필수적임

- 남은 궁금증: 최근에 MAE를 썼다던 Sapiens도 동일한 전략을 채택했을까? 이번 주말 안에 알아보고 싶다
