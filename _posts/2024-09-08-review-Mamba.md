---
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces Review"
layout: post
use_math: true
---

<img src="https://github.com/user-attachments/assets/5a9b4e18-b8bd-4967-93b7-c23edc7ae693" alt="Mamba figure" class="post-pic"/>
<br />
<br />


## Introduction

이 글에선 인트로 요약은 생략(본문이 어려워서 본문 적기도 바쁜지라..)

## State Space Models

- S4(Structured State Space sequence model)은 1차원 함수나 함수열을 N차원 latent space로 보내는 맵이다.(함수열의 경우 시간축 t에 의존)
- S4는 CNN, RNN, 클래식 상태 공간 모델과 연관있다.
- S4는 4가지 파라미터 $(\Delta, \mathbf{A}, \mathbf{B}, \mathbf{C})$와 6가지 식으로 두 단계에 걸쳐 정의된다
    - Continuous version: $h’(t) = \mathbf{A}h(t) + \mathbf{B} x(t), y(t)=\mathbf{C}h(t)$
    - Recurrence after discretization:  $h_t = \bar{\mathbf{A}}h_{t-1} + \bar{\mathbf{B}} x_t, y_t=\mathbf{C}h_t$
    - Convolution after discretization: $\bar{\mathbf{K}} = (\mathbf{C}\bar{\mathbf{B}}, \mathbf{C}\overline{\mathbf{A}\mathbf{B}}, \dots, \mathbf{C}\bar{\mathbf{A}}^k\bar{\mathbf{B}}, \dots), y=x\ast\hat{\mathbf{K}}$

### 이산화

- 첫번째 단계 이산화에선 연속 파라미터 $(\Delta, \mathbf{A}, \mathbf{B})$를 이산 변수 $(\bar{\mathbf{A}}, \bar{\mathbf{B}})$로 *이산화 룰* $(f_A, f_B)$를 통해 $\bar{\mathbf{A}}=f_A(\Delta, \mathbf{A}), \bar{\mathbf{B}}=f_B(\Delta, \mathbf{A}, \mathbf{B})$로 보낸다. 대표적인 이산화 룰로는 ZOH(zero-order hold)가 있다.(여기서도 이거 쓸거임)

\\(\bar{\mathbf{A}} = \exp(\Delta \mathbf{A}), \bar{\mathbf{B}} = (\Delta\mathbf{A})^{-1}(\exp(\Delta \mathbf{A})-I)\cdot \Delta \mathbf{B}\\)

- 이산화는 기본적으론 SSM의 포워드 패스 첫단계이다.
- 이산화는 연속 시스템과 연결성을 만들어줘서, 해상도 불변성을 만들고 normalize를 보장해준다.
- 이산화는 RNN의 gating mechanism과도 연관 있다. 추후 언급

### 계산

이산화로 $(\Delta, \mathbf{A}, \mathbf{B}, \mathbf{C})$가 $(\bar{\mathbf{A}}, \bar{\mathbf{B}}, \mathbf{C})$가 된 이후, 위 식처럼 linear recurrence 또는 global convolution 둘 중 한 방법으로 계산이 가능하다.

- 일반적으론 convolution써서 병렬 학습을 하다가, autoregressive inference를 위해 recurrence 모드로 바꿈
- **왜 convolution이 이산화의 일종**이지? 이것은 [recurrence를 풀어 쓰는 것으로 보일 수 있다](https://huggingface.co/blog/lbourdois/get-on-the-ssm-train).

$h_t$에 대해 풀어쓰면, 

\\( \begin{align\*} h\_0&=\bar{\mathbf{B}}x\_0\\\ 
h\_1&=\bar{\mathbf{A}}h\_0+\bar{\mathbf{B}}x\_1=\bar{\mathbf{A}}\bar{\mathbf{B}}x\_0+\bar{\mathbf{B}}x\_1\\\ 
h\_2&=\bar{\mathbf{A}}h\_1+\bar{\mathbf{B}}x\_2=\bar{\mathbf{A}}^2\bar{\mathbf{B}}x\_0 +\bar{\mathbf{A}}\bar{\mathbf{B}}x\_1+\bar{\mathbf{B}}x\_2\\\ 
h\_3&=\bar{\mathbf{A}}h\_2+\bar{\mathbf{B}}x\_3=\bar{\mathbf{A}}^3\bar{\mathbf{B}}x\_0 +\bar{\mathbf{A}}^2\bar{\mathbf{B}}x\_1+\bar{\mathbf{A}}\bar{\mathbf{B}}x\_2+\bar{\mathbf{B}}x\_3\\\
\vdots \end{align\*} \\)

다시 $y_t$에 대해 풀어쓰면, 

\\(
\begin{align\*} y\_0&=\mathbf{C}\bar{\mathbf{B}}x\_0\\\ 
y\_1&=\mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}}x\_0+\mathbf{C}\bar{\mathbf{B}}x\_1\\\ 
y\_2&=\mathbf{C}\bar{\mathbf{A}}^2\bar{\mathbf{B}}x\_0 +\mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}}x\_1+\mathbf{C}\bar{\mathbf{B}}x\_2\\\ 
y\_3&=\mathbf{C}\bar{\mathbf{A}}^3\bar{\mathbf{B}}x\_0 +\mathbf{C}\bar{\mathbf{A}}^2\bar{\mathbf{B}}x\_1+\mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}}x\_2+\mathbf{C}\bar{\mathbf{B}}x\_3\\\ 
\vdots\\\ 
y\_n&=(\mathbf{C}\bar{\mathbf{B}}, \mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}}, \dots, \mathbf{C}\bar{\mathbf{A}}^n\bar{\mathbf{B}})\ast(x\_0, x\_1, \dots, x\_n) \end{align\*}
\\)

(여기서 convolution은 역순의 내적)

### Linear Time Invariance(LTI)

- 중요한 것은 위 모든 식들에서 이산변수와 연속변수가 전부 시간에 따라 고정이란 것이다. 이를 LTI(linear time invariance)라 하며, recurrence와 convolution 둘 다 연관이 있다.
- 대충 설명하자면 LTI SSM들은 모든 선형 재귀나 선형 convolution과 동일한 취급할 수 있다는 것.
- 지금까진 S4가 LTI였지만, LTI 모델은 특정 데이터 형태에 대해 근본적인 한계가 있다. 이 논문은 그 제약을 극복할 것.

### Structure and Dimensions

- S4는 행렬 $\mathbf{A}\in\mathbb{R}^{N\times N}, \mathbf{B}\in\mathbb{R}^{N\times1},\mathbf{C}\in\mathbb{R}^{N\times1}\$에 구조 제약을 건다. 이 논문에서 쓸 제일 유명한 제약은 대각 행렬일 것.
    - 이 경우 각 행렬 $\mathbf{A}, \mathbf{B}, \mathbf{C}$을 $N$개의 숫자로 표현할 수 있다.
- 인풋이 배치사이즈 $B$, 채널 $D$, 길이 $L$을 가진다고 하면, SSM은 channel-wise하게 적용된다.
- 이 경우 hidden state output은 $DN$개. 시간과 메모리는 $O(BLDN)$→여기서 보틀넥 발생

### 일반적인 SSM

- SSM을 잠재상태가 있는 재귀 프로세스라고 했지만, 더 다양한 형태가 있다.
    - 강화학습의 마르코브 결정 프로세스
    - 계산 신경과학의 다이나믹 캐주얼 모델링
    - 통제이론의 칼만 필터
    - 기계학습의 히든 마르코브 모델, 선형 동역학계
    - 딥러닝의 재귀모델
    - 이 논문에선 보통은 S4를 의미하는 걸로

### SSM 아키텍쳐

- CNN에서 컨볼루션 레이어가 있듯이, SSNN(=SSM 아키텍쳐)에서 SSM 레이어가 있는 것으로 생각한다.
- 알려진 모든 SSM 아키텍쳐
    - 선형 어텐션
    - H3
    - 히예나
    - RetNet
    - RWKV
    - 등등

## Selective State Space Models

우선은 셀렉션 알고리즘의 모티브를 제시하고, 어떻게 이를 SSM에 넣는지 소개하고, 컨볼루션을 계산 못하므로 이를 극복하는 하드웨어-인지 메커니즘을 소개하고, 그 후 가장 단순한 SSM 구조를 소개하겠다. 마지막으로, 이 셀렉션 메커니즘의 특징을 소개한다.

### 1. 모티브: 압축 수단으로서의 셀렉션

- 핵심: 문맥을 어떻게 작은 상태로 압축하는가?
- 이런 관점에서 보면 어텐션은 되기도 하고 안되기도 함;
    - KV cache(?!)를 예시로 생각해보셈. inference 때매 전체 문맥을 통째로 저장해야되잖슴. 이럼 인퍼런스도 느려지고 트랜스포머 학습도 느려짐.
    - (KV cache=[추론속도를 향상시키기 위해 key-value 페어를 저장해두는 것](https://medium.com/@joaolages/kv-caching-explained-276520203249))
    - 물론 상태가 유한하니까 상수시간 인퍼런스, 선형시간 학습을 보장해줌. 근데 이 유한 상태에 효율성이 제약되버림.
- 이 원리를 이해하기 위해서 2가지 인위적 태스크를 고려해보자
![mamba1](https://github.com/user-attachments/assets/cc83c505-a6cc-42f3-96ea-4e458ce16d33)

1. 그냥 카피(인위적 ㄴㄴ): 이렇게 인풋 아웃풋이 필요하면 인풋 내용물을 알 필요가 없으며, 일반적인 재귀나 글로벌 컨볼루션으로 해결됨
2. 선택 카피: 인풋이 선택적으로 랜덤갭을 두고 들어오는 경우. 이러면 모델은 선택적으로 인풋의 내용물을 기억해야한다.
3. 문맥에 따라 다음 올 내용을 알아야하는 인덕션 헤드 태스크. LLM의 핵심 능력이다.
- 이 태스크들이 LTI 모델의 실패를 보여준다.
    - 재귀의 경우, LTI의 파라미터는 상수인데, 문맥과 내용에 따라서 어떻게 인풋을 뽑겠는가?
    - 컨볼루션의 경우, 바닐라 카피는 해결이 되는 것으로 알려져있으나(시간 인지), 이들은 내용물을 알지 못하므로 선택카피 문제는 해결할 수 없다(내용 인지는 불가).
    - 아님 걍 간격이 랜덤하므로, 고정된 컨볼루션 커널로는 계산할 수가 없는 것
- 결국 효율vs효과적 모델의 트레이드오프는 이 상태압축 능력에 의해 결정됨
    - 효율적인 모델은 상태가 작다.
    - 효과적 모델은 모든 문맥 정보를 이해하는 상태를 가지고 있다.
- 결국 이 수열 모델을 결정짓는 것은 **선택성**. 즉 내용물을 인지하는 능력으로 인풋을 집중하거나 거르는 것이 가능.

### 2. Selection SSM
![mamba2](https://github.com/user-attachments/assets/5a9b4e18-b8bd-4967-93b7-c23edc7ae693)

- 원래는 파란 선이 없었음. $B, C, \Delta$ 전부 시간이나 $x$에 의존하지도 않았음. 여기서 인풋의 채널수 $D=5$, latent vector $h_t$의 차원을 $N=4$라고 하자.
- Selective SSM에선 $s_B(x), s_C(x), s_\Delta(x)$를 통해 파라미터 $\mathbf{B}, \mathbf{C}, \Delta$를 인풋에 의존하게 함
- $s_B(x)=\text{Linear}\_N(x), s\_C(x)=\text{Linear}\_N(x), s\_\Delta(x)=\text{Broadcast}\_D(\text{Linear}\_1(x)), \tau_\Delta=\text{softplus}$
    - softplus는 [ReLU의 부드러운 근사치](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html)
- $\mathbf{B}=s_B(x), \mathbf{C}=s_C(x), \Delta=\tau_\Delta(\text{Parameter}+s_\Delta(x))$

### 3. 효율적으로 적용하려면?

- convolution과 attention은 이미 하드웨어(GPU)에 효율적으로 작용한다. Selective SSM도 그렇게 만들고 싶다.
- LTI 모델이 여태 다른 selective model 대신 쓰인 이유: 계산이 좀 효율적이어서

#### 이전 모델들의 문제점

- SSM처럼 되지 않으려면, 히든 스테이트 차원을 최대로 높이고(표현력 증가) 스피드와 메모리 용량을 많이 안 잡아먹어야함
- 재귀 모델이 컨볼루션 모델보다 유연함. 계산 과정만 봐도..하지만 히든 스테이트 차원이 기존에는 너무 높았음. 그래서 기존에 컨볼루션 커널을 써서 스테이트 계산 과정을 생략한 거임.
- 전 LTI 모델들은 둘 다 써서 히든 스테이트 차원도 늘리고 효율성도 챙김

#### 극복 방법: 하드웨어를 인지한 스테이트 확장

- 관찰할 점 2개:
    - 그냥 재귀 계산을 하면 $O(BLDN)$이 걸리고 컨볼루션은 $O(BLD\log(L)$이 걸림. 수열이 길고 스테이트 차원이 그리 높지 않다면, 재귀가 나음.
    - 문제 2개는 재귀의 수열적 특성과 과한 메모리 사용량임. 메모리 사용량은 좀 아끼기 위해서, 히든스테이트 전체를 다 메모리로 구현하고 싶지는 않음
- GPU 서열을 통해 메모리를 효율적으로 쓰고 싶음. 특히 모든 연산(행렬곱 등)이 메모리 대역폭에 의해 제한되어있음.
- 스캔 작업+통합 커널: GPU HRAM(고대역폭 메모리)에 $(B, L, D, N)$크기의 스캔 인풋 $\bar{\mathbf{A}}, \bar{\mathbf{B}}$을 넣지 말고,  SSM 파라미터 $(\Delta, \mathbf{A}, \mathbf{B}, \mathbf{C})$를 느린 HBM에서 빠른 SRAM으로 직접 넣고, 거기서 이산화, 재귀 작업을 해서 최종 아웃풋을 HBM으로 꺼내온다.
    - 수열이 SRAM에서 계산하기에 너무 긴 경우엔 여러 청크로 나누어 계산한다.
    - 이 방법을 쓰면 20~40배 정도 계산이 빨라진다.
- 재귀의 수열적 특성을 피해야 계산이 효율적이다. 비선형적이긴 하지만, 병렬 스캔 알고리즘이 알려져 있으므로 이걸 사용한다.
- 또 백프롭에 쓰이는 중간 스테이트 저장을 피하고 싶다. 재계산을 통해 중간 상태는 저장하지 않고, 인풋을 HBM에서 SRAM으로 올릴 때 백워드 패스에서 재계산된다.

### 4. 단순한 SSM 구조

- 이제 이렇게 만든 Selective SSM은 어느 뉴럴 네트워크든 넣을 수 있다.
- 우린 H3와 gated-MLP를 참조해 다음과 같이 네트워크를 구성했다.
![mamba3](https://github.com/user-attachments/assets/278d055c-5d52-41f0-a76a-584c09178445)

- 이 블록을 반복함
- SiLU(Swish activation) 사용

### 5. 셀렉션 메커니즘의 특징

- 셀렉션은 사실 다른 모델에 쓰인 것으로 볼 수 있으며, 다른 파라미터에 적용하거나, 다른 트랜스포메이션을 쓰는 것으로 이해할 수 있다.
- RNN의 게이팅 메커니즘: 다음과 같은 정리가 성립한다.

>$N=1, A=-1, B=1, s_\Delta=\text{Linear}(x), \tau_\Delta=\text{softplus}$라고 가정하자. 그러면 셀렉티브 SSM은 다음 식이 된다.

\\(
\begin{align*} g_t &= \sigma({\text{Linear}(x\_t)) \\\ 
h\_t &= (1-g\_t)h\_{t-1}+ g\_t x\_t \end{align*}
\\)

- 셀렉션 메커니즘은 다음과 같은 3가지 효과를 가진다:
    - 변수에 간격을 만들어줌: 셀렉션이 노이즈 토큰을 제거해주고 관심있는 데이터 인풋만 받게 해줌.
    - 문맥을 필터링 해줌: 대부분의 수열 모델이 이상하게도 긴 문맥에 약함. 원인은 관련 없는 문맥을 걸러내는 능력이 없어서인 것으로 추정됨. 셀렉션 모델은 언제든지 스테이트를 리셋할 수 있어서 긴 문맥 길이를 가질 수록 좋은 성능을 냄
    - 경계를 리셋해줌: 여러 독립 수열이 들어올 때, 트랜스포머는 어텐션 마스크를 통해 얘네를 잘 분리하지만, 일반적인 수열 모델은 구분을 잘 못함. 셀렉션 모델은 경계에서 이 스테이트를 리셋할 수 있음.
- 셀렉션 파라미터 3가지 각각의 효과에 대해 다음 해석이 있다:
    - $\Delta$의 해석: $\Delta$는 인풋을 얼마나 집중하거나 무시할지 통제하는 역할을 한다. RNN 게이트를 생각하면 이해가 빠를 것. 또 SSM에선 연속 시스템을 이산화하는 타임스텝 역할을 하는데, 타임스텝이 길어져서 무한에 가까워지는 것은 현재 인풋에 더 길게 집중하는 것이고, 짧아져서 0에 가까워지는 것은 현재 인풋을 무시하는 것이라고 직관적으로 이해할 수 있다.
    - $\mathbf{A}$의 해석: $\mathbf{A}$는 $\Delta$를 통해서만 모델에 영향을 미치므로, 굳이 곱해지는 애들을 둘 다 셀렉티브하게 만들지는 않았다. 해봤는데 딱히 성능 차이도 없음.
    - $\mathbf{B}, \mathbf{C}$의 해석: 인풋을 히든스테이트로 넣을지 말지, 히든 스테이트를 아웃풋으로 얼마나 넣을지 섬세한 컨트롤이 가능해짐. 모델이 재귀 다이나믹스를 내용물(입력)과 문맥(히든 스테이트)에 기반해서 통제하는 것으로 이해할 수 있다.

### 6. 모델 디테일

- 복소수를 자주 쓰긴 하는데 실수가 더 좋거나 차이가 없는 경우도 있어서, 히든 스테이트에 실수 사용
- 초기화는 S4D-Lin과 S4D-Real 사용함. 히포 이론에 따른 것.
- $\Delta$ 매개변수화 할 때 1차원에 박고 브로드캐스팅 했지만, 더 큰 차원에도 넣을 수 있고 여러 해석이 가능

## 실험 결과

- 맘바모델이던 S6 블록이던, 블록을 쓰기만 하면 Selective copying 정확도에서 99%를 넘는 압승. 설계가 당연히 인풋의 랜덤성을 반영하니까..
- 인덕션헤드 태스크에서도 매우 긴 수열에 대해 맘바가 유지력 압승.
- 스케일링 법칙에서도 맘바가 너무 좋았으며, 심지어 어텐션 쓰는 트랜스포머++(라마)에 대해서도 동등한 perplexity를 자랑함
    - perplexity: [언어모델 평가하는 지표](https://wikidocs.net/21697). 로그 퍼플렉시티면 문장에서 각 단어가 나올 확률에 로그를 씌운 평균
- 제로샷 평가에서도 최강
- DNA 모델링에서도 최강
- 유넷 백본을 쓰는 오디오 모델링, 오디오 생성에서도 최강

## 문제점

- 연속자료형(오디오, 비디오)에서 SSM들이 강한 인덕티브 바이어스를 가지는데, selective ssm은 이산 자료형(텍스트, DNA)에 대한 약점을 극복했으나 연속 자료형에 대한 성능을 보장받지 못한다.
- 실험을 작은 모델 사이즈에서만 진행함
