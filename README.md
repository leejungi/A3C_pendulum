# A3C_pendulum

## 목차 

1. [Asynchronous Advantage Actor Critic](#asynchronous-advantage-actor-critic)  
1. [N-step TD](#n-step-td)  
1. [Entropy](#entropy)  
1. [Algorithm](#algorithm)
1. [Reference](#reference)  

## Asynchronous Advantage Actor Critic

Replay buffer를 사용함으로써 memory와 real interaction마다 computation을 요구하는 문제를 해결하기 위해 asynchronous advantage actor critic 방법을 제안. 이전에 비슷한 방법이 있지만 이 논문에서는 한 cpu에서 multi thread를 이용하는 방법을 사용. one-step Sarsa, one-step Q learning, n-step Q-learning, advantage actor critic 모두를 사용할 수 있음. sarsa의 문제점인 많은 실험 trial이 필요한 것을 multi thread를 통하여 실험 시간을 줄일 수 있음. 그리고 여러 환경에서 학습하기 때문에 model robust하고 stable하다.

global network를 두고 각 thread마다 policy를 둔다. 각 thread 마다 실험 환경을 생성하여 policy를 업데이트를 하고 일정 episode를 학습한 후 global network에 gradient를 전달해준다. global network를 해당 gradient로 업데이트 후 global network를 이용하여 thread policy를 업데이트한다. 

## N-step TD

N=3일 시 t=0에서 3-step TD를 적용하고 t=1에서 2-step TD를 적용한다. 

>$$V_0 \to V_1 \to V_2 \to V_3$$

when t=0

>$$A_0 \triangleq R_0 + \gamma R_1 + \gamma^2 R_2 + \gamma^3 V(3) - V(0)$$

## Entropy

disbribution이 커질 수록 불확실성이 높아진다.

>$$H(p(x) \triangleq \int -p(x) \text{ln}p(x) \, dx$$
	
>$$ = \int^\infty_{-\infty} -p(x) \text{ln} \frac{1}{\sqrt{2\pi \delta^2}} \text{e}^{-\frac{(x-\mu)^2}{2\delta^2}} \, dx$$

>$$=\int^\infty_{-\infty} p(x) \text{ln} \sqrt{2\pi \delta^2} \, dx + \int^\infty_{-\infty} p(x) \frac{(x-\mu)^2}{2\delta^2}\, dx$$

$ =\frac{1}{2} (1 + \text{ln} 2\pi \delta^2) $

>$$E[(x-\mu)^2]=\delta^2$$

entropy는 분산에 비례한다.

Entropy가 높아진다는 뜻은 exploration 가능성이 높다는 뜻

논문에서는 이 term을 못 찾았음




## Algorithm

![algorithm][Algorithm]  
_A3C algorithm_

## Code

주의사항  
1. Cuda에서 multiprocessing을 하기 위해서는 spawn환경 설정이 필수  
2. Window에서cuda multirprocessing이 아직 호환안되는 것 같다. 확인 필요


## Reference
1. [혁펜하임 유투브][혁펜하임 유투브]  
2. [Mnih, Volodymyr, et al. "Asynchronous methods for deep reinforcement learning." International conference on machine learning. PMLR, 2016.][paper]

[혁펜하임 유투브]: https://www.youtube.com/watch?v=cvctS4xWSaU&list=PL_iJu012NOxehE8fdF9me4TLfbdv3ZW8g  
[paper]: https://arxiv.org/pdf/1602.01783.pdf

[Algorithm]: /assets/algorithm.png

