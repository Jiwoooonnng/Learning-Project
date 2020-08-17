import numpy as np
import gym
from matplotlib import pyplot
import matplotlib.pyplot as plt
from collections import defaultdict

plt.style.use('ggplot')


######## 블랙잭 환경 생성 ########
env = gym.make('Blackjack-v0')
#env.action_space, env.observation_space

#### < 보상 > ####
# 플레이어가 21일 경우 : +1
# 플레이어가 질 경우 : -1
# 플레이어가 비길 경우 : 0

#### < 행동 > ####
# hit : 카드를 받음
# stand : 카드를 받지 않음

#### < 상태 > ####
# [(a,b,c)]의 리스트 내 튜플 형태
# a : 내 카드 2개의 숫자 합
# b : 딜러 카드의 숫자
# c : ace의 유무

#### 게임 방법 ####
# 1. 카드 합이 먼저 21이되거나 딜러보다 높으면 승리
# 2. 카드를 추가하거나(hit) 안 받을 수 있음(stay)
# 3. 카드 추가 시 21을 넘으면 패배(burst)
# 4. K, Q, J는 10, Ace는 1 or 11로 선택가능


#### 환경 오류 ####
# 1. env.step 적용 시 상대의 카드 숫자가 변화가 없음
# 2. 플레이어가 21이 아닐경우 항상 -1반환


######## 함수 생성 ########

# 샘플 정책 함수
def sample_policy(observation) :
    # 20 이상이면 stand(0)
    # 20 미만이면 hit(1)
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

# 에피소드 생성 함수
def generate_episode(policy, env) :
    states, actions, rewards = [], [], []

    # 랜덤으로 상태, 행동, ace 유무 생성
    observation = env.reset()
    while True :
        # observation 변수 추가
        states.append(observation)

        # policy 함수를 통해 행동 선택
        action = policy(observation)
        actions.append(action)

        # 현재 가능한 행동에 대해 다음 환경을 랜덤으로 생성
        # done : 최종 상태(21 이) 도달 여부 반환
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        if done :
            break

    return states, actions, rewards

# 일회 방문 몬테카를로 예측 함수
def first_visit_mc_predict(policy, env, n_episode) :
    # 변수 초기화
    value_table = defaultdict(float)
    N = defaultdict(int)

    # n_episode 만큼 반복 학습
    for k in range(n_episode) :
        states, _, rewards = generate_episode(policy, env)
        returns = 0

        # states마다 반복
        for t in range(len(states)-1,-1,-1) :
            R = rewards[t]
            S = states[t]
            returns += R

            # 1회 방문인지 확인(처음 방문이면 value 추가)
            if S not in states[:t] :
                N[S] += 1
                value_table[S] += (returns - value_table[S]) / N[S]
    return value_table

# 그래프 그리기
def plot_blackjack(value_table, ax1, ax2) :
    player_sum = np.arange(12, 21+1)
    dealer_show = np.arange(1,10+1)
    usable_ace = np.array([False, True])

    state_values = np.zeros((len(player_sum),
                            len(dealer_show),
                            len(usable_ace)))

    for i, player in enumerate(player_sum) :
        for j, dealer in enumerate(dealer_show) :
            for k, ace in enumerate(usable_ace) :
                state_values[i,j,k] = value_table[player, dealer, ace]

    X, Y = np.meshgrid(player_sum, dealer_show)

    ax1.plot_wireframe(X, Y, state_values[:, :, 0])
    ax2.plot_wireframe(X, Y, state_values[:, :, 1])
    for ax in ax1, ax2 :
        ax.set_zlim(-1,1)
        ax.set_ylabel('Player sum')
        ax.set_xlabel('dealer show')
        ax.set_zlabel('state-value')

######## 실행 ########

n_episode = 50000
value_table = first_visit_mc_predict(sample_policy, env, n_episode)

fig, axes = pyplot.subplots(nrows=2, figsize = (5,8), subplot_kw={'projection':'3d'})
axes[0].set_title('Value function without usable ace')
axes[1].set_title('value function with usable ace')
plot_blackjack(value_table, axes[0],axes[1])

plt.show()
