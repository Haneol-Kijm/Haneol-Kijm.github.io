---
title: How ChatGPT Works 후기
layout: post
---


## [How ChatGPT Works](https://www.assemblyai.com/blog/how-chatgpt-actually-works/)

- 기술의 발전으로 다음 단어 맞추기 작업을 훌륭히 수행하는 LLM의 개발에 성공했다. 근데 우리가 원하는 건 좀 더 고차원의 인지 작업인데? (Misalignment 발생)
- 조정을 하지 않은 LLM의 문제
    - 지시를 따르지 않는다
    - 헛소리를 한다(환각 현상)
    - 결론에 도달한 논리 해석이 불가능하다
    - 편향되거나 안좋은 답변을 한다
- 첫 문단을 읽고 느낀 점: 왜 지난 번에 안 읽은 내용인 것 같이 느껴지는 걸까. RLHF 글을 읽고 읽으니 이 misalignment 문제가 RLHF와 밀접한 관련이 있는 것이 느껴진다.
- LLM이 보통 학습하는 태스크의 기반은 ‘다음 단어 맞추기’와 ‘빈칸 채우기’다. 그럼 이것만 연습한 LLM이 고차원 논리를 이해하지 못하는 것은 당연하잖아? 대체 ChatGPT 개발진들은 이걸 어떻게 메꾼 건지 궁금해지는 부분
- ChatGPT의 RLHF는 사실 RLHF 뿐만이 아니라 SFT를 동반한다(Supervised Fine-tuning). 다음 세 스텝을 거친다.
    1. 미리 학습된 LLM이 라벨러들이 만든 소량의 demonstration data를 학습한다.(Fine-tune). 이렇게 나온 모델을 SFT라고 하며, 1번 스텝은 한 번만 시행
    2. SFT로부터 대량의 output을 뽑아낸 다음, 라벨러들이 여기에 대해 대량의 비교 답안 데이터를 만든다.(RLHF 글로부터 읽은 것에 따르면 ‘선택지 선호도 조사’일 확률이 높다). 이 데이터만 가지고 새로운 보상 모델(아마 preference model)을 만든다.
    3. preference model이 SFT를 fine-tune한다. 그리고 23번을 무한반복. 아마 아래 사진이 될 확률이 높겠지. 이 결과로 나오는 모델 = Policy model
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ac84168d-557f-4919-b37f-2632c6456077/85b02511-83e7-4ea5-b1c2-0642ad57ca82/image.png)
    
- 1번 스텝의 demonstration data는 이런 식이다. 정해진 질문에 대해 예상되는 답변 목록. 사람이 일일히 적으니 리소스가 많이 듦. 일부는 라벨러가 적고, 다른 일부는 GPT-3을 이용한 고객들한테 수집함(총 약 12000~15000개)
- 의외로 순수한 언어모델이 아니라, 코딩 써주는 모델을 기반으로 정함
- 2번에선 예상대로 라벨러들이 SFT의 output에 대해 선호도 랭킹을 매긴 데이터를 만듦. 그걸로 새로 보상 모델을 만듦(1번 스텝의 약 10배 크기 데이터 사용)
- 최종적으론 Proximal Policy Optimization(PPO)로 SFT를 fine-tune
- PPO의 특징
    - on-policy, 즉 action과 reward에 대해 실시간으로 업데이트를 진행하고 피드백을 받음
    - trust-region optim을 통해 너무 큰 업데이트가 일어나는 것을 막음(불안정해지지 않도록)
    - 저번에도 이해가 막혀서 포기한 부분이 이 부분인데, 이를 이해하려면 **RL 자체와 PPO에 대해 따로 공부**할 필요가 있어 보인다.
    value function→advantage func 계산=expected_return-current_return
    adv_func→policy update by 비교: current policy의 액션과 이전 policy가 했을 action
    그냥 자명한 무언가를 하는 것 같은 느낌, 그리고 이게 왜 다른 알고리즘과 비교해서 더 좋은 점이 되는지를 모르겠음.
    
    > PPO uses a **value function to estimate the expected return of a given state or action**. The value function is used to compute the advantage function, which represents the difference between the expected return and the current return. The advantage function is then used to update the policy by comparing the action taken by the current policy to the action that would have been taken by the previous policy. This allows PPO to make more informed updates to the policy based on the estimated value of the actions being taken.
    > 
- 아무튼 최초 PPO는 SFT, 최초 val_func은 reward model이 함. 이건 결국 bandit process가 되는데 이건 들어본 적이 있음(일종의 룰렛 기계)
- 쨌든 일련의 과정을 거치면 평가가 남는데, 라벨러들한테 시키면 과편향되니까 고객들의 답변을 테스트셋으로 사용
- 평가기준: 도움이 되었는가? 믿을 만한가? 해가 안 되는가?
- 최종 PPO모델이 원본 모델로 성능이 돌아와버리는 현상이 관측되는데, 이를 해결하기 위해 믹스 경사하강법을 사용: SFT와 PPO의 기울기를 스까서 사용
- 이 모델 학습방법의 한계점
    - 라벨러들의 취향도 문제, 라벨러들한테 제시할 기준 잡아주는 연구자들도 문제, 고객들이 만든 답변을 취합할 때도 문제, 라벨러 편향이 학습과 평가에 둘 다 들어가는 것도 문제
    - 결국 라벨러들과 연구자들이 최종 고객들의 취향을 대변할 수 있어?
    - 통제 연구의 부족: 최종 PPO를 왜 SFT랑 비교하는 거야? RLHF 없이 다량의 데이터를 박아서 학습한 SFT랑 비교해야지→결국 그럼 RLHF가 잘 해주는 게 맞는 건지 의문
    - 비교 데이터의 ground truth 부족: 라벨러들의 선호도가 일치한다는 보장이 있어?
    - 인간의 취향은 유사성을 갖지 않아
    - 보상 모델이 응답을 일정하게 뱉는다는 연구도 없어: 만약 말만 다르고 같은 내용인 두 질문을 모델에게 줬을 때, 결과가 같을까?
    - 강화학습 관점에서의 과적합 이슈: ChatGPT는 이를 방지하기 위해 보상 함수에 KL-페널티를 부여
- 다 읽은 후기: 지식이 없어도 읽을 수 있게 써놨다고 해놓고 강화학습에 대한 지식이 없으면 전부 이해는 못하게 돼있어서 읽기 힘든 부분들이 있었다.
다만 결국 이해는 다 했고, 강화학습 기초도 좀 읽어볼 필요가 있음을 체감한다.
