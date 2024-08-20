---
title: How RLHF Works 후기
layout: post
---

## [How RLHF Works](https://www.assemblyai.com/blog/how-rlhf-preference-model-tuning-works-and-how-things-may-go-wrong/)

- 서문을 보며 드는 생각: 생성형 모델은 비전 분야던 언어 분야던 사람의 통제(Control)이 필요하다는 점이다. 최근 트위터와 허깅페이스에서도 [디퓨전 이미지 생성을 통제하는 방법](https://huggingface.co/papers/2408.06070)에 대한 논의가 있는데, 인류는 강력한 생성툴을 얻은 대신 랜덤성을 아직 통제하지 못하고 있는 모양이 아닐까 싶다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ac84168d-557f-4919-b37f-2632c6456077/85b02511-83e7-4ea5-b1c2-0642ad57ca82/image.png)
    
- RLHF란 선호 모델을 통해 이미 훈련된 기저 (언어) 모델의 답변에 선호도를 매겨, Fine-Tuning 하는 강화학습이다. 굳이 선호 모델이 아닌 보상 모델들이 존재는 하나, 비용 문제로 선호 모델이 쓰임
- 위를 보고 느낀 점: **다른 분야의 학습 방법을 다양하게 알고 있는 것이 점점 중요하단** 생각이 든다. 비전 트랜스포머도 언어 모델에서 온 방법론이며, ChatGPT도 언어모델에 강화학습 방법론을 추가로 적용시킨 것이며, 뛰어난 연구자나 기술자라면 이런 방법론의 Mix를 할 줄 알아야 한다고 본다.
- RLHF의 해석: 혼돈 앵무새를 인간 기호에 맞추게 통제하면 답변이 정형화되고 다양성이 제한된다. 이는 이득인가 손해인가? 만약 목적이 정확성을 요하는 보조, 검색분야라면, 이득이다. 그러나 아이디어 만들기나 글쓰기 보조 목적으로 쓴다면, 창의성을 죽이는 방향이 되버린다.
- RLHF의 한계
    - **환각** 현상을 부추길 수 있다. RLHF를 추가했는데 오히려 거짓 정보나 창조된 정보를 답변하는 오류가 생기는 것이다. 또 신기한 점으론, 인간 평가자들이 평가에서 환각 정보를 좀 더 선호하는 경향이 있다는 것이다.
    - 이는 또다른 문제점을 제시한다: 애초에 사람이 모델 답변의 우열함을 가리는 것이 어렵다…진실성을 가리는 것이 주관적일 수 밖에 없는 것
    - 객관적인 지표를 도입한다면 어떨까? 이에 대한 반례로, 극단적인 답변으로 조정된 모델이 진실성 지표 점수를 높게 기록한 실험도 존재한다.
    - RLHF는 탈옥 공격에도 취약하다. 동사 하나를 바꾸는 것만으로 안정성에 문제가 생긴다.
    - 언어를 바꾸면 직업 성편견이 생긴다던가 하는 이상한 경향도 갖고 있다.
    - RLHF를 개선시켜도 [문제가 해결된다는 보장도 없다](https://arxiv.org/abs/2209.15259?ref=assemblyai.com). 그래서 그냥 이론적으로 불가능한 거 아니냐? 하는 의견들이 존재한다.
    - 비교 연구가 부족한 문제도 있다. 실례로, 단 1000개의 고퀄 데이터셋 사용 만으로 획기적으로 인간 평가 점수를 올린 연구도 있다. 앞으로 높은 데이터 퀄리티vs확장성, 무엇을 우선시 해야할지가 연구 과제로 남아있을 것이다.
- 추가로 읽을 거리
    - [**RLAIF**](https://www.assemblyai.com/blog/how-reinforcement-learning-from-ai-feedback-works/)
    - [**How physics advanced Generative AI**](https://www.assemblyai.com/blog/how-physics-advanced-generative-ai/?utm_source=google&utm_medium=cpc&utm_campaign=brand)
- 느낀 점: 지난 ChatGPT 글보다 훨씬 글이 이해하기 쉬웠고, 다시 ChatGPT 원리 쪽 글 읽는 걸 도전해봐야겠다. RLHF가 아직까지 한계가 많다는 것도 알았고, 그럼 **RLAIF는 뭐고 이건 더 좋은걸까?** 하는 의문이 남는다.
