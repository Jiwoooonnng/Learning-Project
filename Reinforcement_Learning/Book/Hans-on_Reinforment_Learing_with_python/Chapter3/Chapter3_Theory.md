Chapter 3. MDP and DP
====================
1. Markov Chain and Markov Process
---------------------------------
* Markov Property : 미래 상태가 현재 상태에만 의존
* Markov Chain : 오로지 현재 상태에만 의존하여 다음 상태를 예측하는 확률 모델

> 용어
> * Transition : 한 상태에서 다른 상태로 옮겨가는 것
> * Transition Probability : 다른 상태로 옮겨갈 확률

2. MDP(Markov Decision Process)
---------------------------------
A. Value and Return
>  * Value(r) : 수행한 행동의 보상
>  * Return(R) : 환경으로 받느 미래 보상의 총합
>    R_{t} = r_{t+1} + r_{t+2} + ... + r_{T}
