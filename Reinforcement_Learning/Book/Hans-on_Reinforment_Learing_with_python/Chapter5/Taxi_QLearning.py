import gym
import random

######## < 환경 생성 > ########
env = gym.make("Taxi-v3")

#### < 목표 > ####
# R, G, Y, B의 한 지점에서 다른 지점으로 이동
# 되도록 빠른 시간 내 정확한 지점으로 이동

#### < 상태 > ####
#
#    +---------+
#    |R: | : :G|
#    | : | : : |
#    | : : : : |
#    | | : | : |
#    |Y| : |B: |
#    +---------+
#
# R, G, Y, B : 지점개
# | : 지나갈 수 없는 벽
# : : 지나갈 수 있는 경


#### < 행동 > ####
# 위, 아래, 왼쪽, 오른쪽 한 칸 씩 총 4

#### < 보상 > ####
# 매 스텝마다 -1점 누적
# 정확한 지점에 도착 시 +20
# 잘못된 지점에 도착 시 -10로


#### < 환경 설명 > ####
# 1. 각 에피소드 마다 지도에 랜덤으로 택시가 생성
# 2. 택시는 파란색 지점에서 탑승자를 탑승 시킴
# 3. 택시는 빨간색 지점에서 탑승자를 하차 시킴





######## < 변수 생성 > ########
# learning rate 할당
alpha = 0.4
# discount factor 할당
gamma = 0.999
# epsilon-greedy policy를 위한 epsilon 할당
epsilon = 0.017
# q dictionary 생성 후 상태 및 행동에 대해 0으로 초기화
q = {}
for s in range(env.observation_space.n) :
    for a in range(env.action_space.n) :
        q[(s,a)] = 0

######## < 함수 생성 > ########
# q-table 업데이트
# q-learing 방식으로 업데이트
def update_q_table(prev_state, action, reward, next_state, alpha, gamma) :
    qa = max(q[(next_state, a)] for a in range(env.action_space.n))
    q[(prev_state, action)] += alpha * (reward + gamma * qa - q[(prev_state, action)])

# epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon) :
    if random.uniform(0,1) < epsilon :
        return env.action_space.sample()
    else :
        return max(list(range(env.action_space.n)), key = lambda x : q[(state,x)])

for i in range(8000) :
    # 보상 초기화
    r = 0

    # 초기 상태 랜덤 생성
    prev_state = env.reset()
    print(" ")
    print("-----------")
    print("new episode #", i)
    while True :
        # 환경 보여주기
        env.render()

        # E-G를 이용해 현재 상태의 현재 행동 선택
        action = epsilon_greedy_policy(prev_state, epsilon)

        # 선택된 행동으로 다음 상태 반환
        next_state, reward, done, _ = env.step(action)

        # Q-Learning 업데이트를 이용해 현재 상태, 행동의 q값 업데이트
        update_q_table(prev_state, action, reward, next_state, alpha, gamma)

        # 이전 상태를 다음 상태로 변경
        prev_state = next_state

        # 보상 추가
        r += reward

        # 마지막 상태일 경우 종료
        if done :
            break
    print("total reward : ", r)
env.close()
