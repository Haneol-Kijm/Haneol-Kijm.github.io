---
title: 어떻게 하면 AI code assistant를 무료로 써먹을 수 있을까?
layout: post
---



부스트캠프를 다니면서 프로젝트를 시작하게 되었고, 프로젝트를 하는 와중에도 다른 외부 대회나 행사에 참가하고 싶어졌다.

짧은 시간 안에 이런 대회에 참여하거나 코딩을 짜려면 많은 양의 수작업이 필요한데, 요즘은 ai 시대다. [이런 트위터 영상](https://x.com/yacineMTB/status/1825033947333468363)만 봐도 그냥 ai로 전반적인 뼈대는 빠르게 코딩해놓고 검수하는 편이 낫지 않은가?

그래서 AI 언어모델을 써서 코딩 전반을 보조하는 도구 옵션 들을 한 번 살펴보았다.

## 1. 그냥 코드 자체를 복사-붙여넣기 해서 chatgpt나 gemini한테 직접 물어본다.

이 경우 코드를 짜주거나 갈아끼워주는 것은 가능하지만, 실시간으로 코딩 도중에 뒤에 옵션을 추천해주던가 하는 것은 어렵다.

다만 간간히는 사용할 만한 방법이라고 생각된다.

## 2. 코드 보조 프로그램들을 사용한다.

찾아본 결과, 코드 보조 프로그램(유료)는 다음과 같았다.

- microsoft copilot
- cursor: 특정 사용량 이후 유료 요금 전환 및 청구
- google gemini code asisstant: 24년 11월 며칠까지는 무료라는 것 같긴 한데, 어쨌거나 이후는 유료
- codegpt: 특정 사용량 이후 유료

과연 무료옵션은 정말 없는건가? 하던 와중에 같은 조 조원분께서 이런 무료 옵션이 있다는 걸 찾아서 검색해주었다.

## 3. 무료 코드 보조 프로그램

일단 내가 처음 찾았던 무료 옵션은 [dingllm](https://github.com/yacineMTB/dingllm.nvim/tree/master)이었다. 다만 이 스크립트는 neovim에서만 적용가능하다는 단점이 있다.

조원분이 알려주신 옵션은 [continue](https://www.continue.dev/)이다. 컨티뉴는 VSCode에 깔아서 쓸 수 있는 확장프로그램으로, ai 기반의 코딩을 보조해준다. 

문제는 이 프로그램을 사용하려면 AI 언어모델인 LLM을 내가 추가로 구해와야 한다는 것이다.

### 옵션 1. 직접 로컬에 LLM 모델을 설치한다.

이 옵션은 힘든게, 내 노트북이 오래되기도 했고, 용량이 부족하다.

### 옵션 2. 서버에서 무료로 제공해주는 모델을 찾는다.

dingllm의 config를 확인해 본 결과, 다음과 같은 서버 api 제공자들이 있는 것을 확인할 수 있었다. dingllm을 안써봐서 모르겠지만, 이걸 돌아가며 쓰는 것으로 api 부담을 줄이는 것처럼 보인다…

- LLaMA 3.1
    - openrouter.ai
    - groq
    - lambdalabs
- Claude 3.5
    - anthropic←이건 코드에서 실제로 사용 안되는 듯?

일단 제공자들을 찾았으니, 걱정없이 VSCode에 일단 continue를 깔아보는게 좋아보였다.

컨티뉴 문서에서 소개된 컨티뉴의 기능들:

1. [Understand code sections easily](https://docs.continue.dev/how-to-use-continue#easily-understand-code-sections): 드래그한 부분을 설명해줘~라고 하면 설명해줌
2. [Autocomplete code with tab](https://docs.continue.dev/how-to-use-continue#tab-to-autocomplete-code-suggestions): 유명한 기능. 구글 코랩에서도 쓸 수 있음
3. [Refactor functions in-place](https://docs.continue.dev/how-to-use-continue#refactor-functions-where-you-are-coding): 드래그해서 ‘이 함수를 재귀함수로 바꿔줘’라고 하면 바꿔줌
4. [Query your codebase](https://docs.continue.dev/how-to-use-continue#ask-questions-about-your-codebase): @codebase를 앞에 붙이면 내 코드에 대해 아무거나 물어볼 수 있음
5. [Leverage documentation as context](https://docs.continue.dev/how-to-use-continue#quickly-use-documentation-as-context): 포크한 레포지토리의 문서를 읽기 귀찮을때 빠르게 찾아줌
6. [Use slash commands for quick actions](https://docs.continue.dev/how-to-use-continue#kick-off-actions-with-slash-commands): 
7. [Add various elements to context](https://docs.continue.dev/how-to-use-continue#add-classes-files-and-more-to-context)
8. [Instantly understand terminal errors](https://docs.continue.dev/how-to-use-continue#understand-terminal-errors-immediately): 에러를 바로 이해 가능

그냥 링크를 직접 타고 가서 움짤을 보면 이해할 수 있을 정도로 친절하게 문서화가 잘 되어 있으니 직접 읽어보시는 편을 추천

설치는 매우 간단했다. VSCode extensions 탭에서 그냥 설치하면 끝

다만 내가 현재 오른쪽 손목 건강을 위해서 상하좌우 키를 ctrl+IKJL로 고쳐쓰고 있는데, 단축키 CTRL+L이 겹쳐서 이 부분만 수정해주었다.

채팅하거나 코드를 분석하는 것까진 LLaMA로 충분히 잘 되는 것 같다.

다만 코드를 자동으로 추천해주는 autocomplete 기능은 만족스러운 성능이 안나오는 것 같은데, 이 부분은 코드 모델을 따로 찾아서 써야할 것 같다.

디폴트로 추천해주는 codestral 이건 유료 모델인 것을 확인해서, 무료모델을 내일 추가로 찾아봐야겠다…
