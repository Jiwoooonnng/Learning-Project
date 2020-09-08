import numpy as np
import gym
import tensorflow as tf
from tensorflow.contrib.layers import flatten, conv2d, fully_connected
from collections import deque, Counter
from datetime import datetime

env = gym.make("MsPacman-v0")
n_outputs = env.action_space.n

######## 이미지 전처리 ########
#### 이미지 기본 ####
# img = [행,열,channel]
# img 값은 0 ~ 255의 int 형
# img의 channel은 R, G, B 3개

# color 평균
color = np.array([210, 164, 74]).mean()

# 이미지 crop, convert, resize
def preprocess_observation(obs) :
    # 이미지 crop 및 resize
    img = obs[1:176:2, ::2]     # [1부터 176까지 2의 간격으로, 2의 간격으로 처음부터 마지막까지]

    # 이미지 channel 부분을 없앰(gray scale)
    img = img.mean(axis=2)

    # 이미지 대비 향상
    img[img==color] = 0         # color하고 값이 같은 arr의 원소들은 0이 됨

    # 이미지 색 크기를 normalize(-1 to 1)
    img = (img - 128) / 128 - 1

    return img.reshape(88,80,1)


######## DQN 정의 ########
#### 네트워크 구성 ####
# Conv2d => Conv2d => Conv2d => Fully_Connected_Layer => Fully_Connected_Layer

tf.compat.v1.reset_default_graph()        # 그래프(계산 과정 저장 객체) 초기화

def q_network(X, name_scope) :
    initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.compat.v1.variable_scope(name_scope) as scope :   # with 문의 경우 자동으로 close를 해줌
        # Convolution X 3
        layer_1 = conv2d(X, num_outputs=32, kernel_size=(8, 8), stride=4, padding='SAME', weights_initializer=initializer)
        tf.compat.v1.summary.histogram('layer_1',layer_1)
        layer_2 = conv2d(layer_1, num_outputs=64, kernel_size=(4, 4), stride=2, padding='SAME', weights_initializer=initializer)
        tf.compat.v1.summary.histogram('layer_2', layer_2)
        layer_3 = conv2d(layer_2, num_outputs=64, kernel_size=(3, 3), stride=1, padding='SAME', weights_initializer=initializer)
        tf.compat.v1.summary.histogram('layer_3', layer_3)
        flat = flatten(layer_3)

        # Fully Connected X 2
        fc = fully_connected(flat, num_outputs=128, weights_initializer=initializer)
        tf.compat.v1.summary.histogram('fc', fc)
        output = fully_connected(fc, num_outputs=n_outputs, activation_fn=None, weights_initializer=initializer)
        tf.compat.v1.summary.histogram('output', output)

        # vars를 dic
        vars = {v.name[len(scope.name):]: v for v in tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
        #### collection ####
        # 변수들을 collection에 그룹화 하여 관리함
        # tf.get_collection(key)으로 key에 해당하는 변수를 불러올 수 있음
        # tf.GraphKey : 많이 사용되는 key 값을 저장한 함수
        # tf.GraphKeys.TRAINABLE_VARIABLES : 변화가능한 변수를 저장한 collection의 키
        return vars, output

######## Decaying Epsilon Greed Policy ########
#### 정의 ####
# 시간이 지날수록 탐험이 줄고 탐욕 정책이 증가

#### 변수 할당 ####
epsilon = 0.5
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 500000

#### 함수 생성 ####
def epsilon_greedy(action, step) :
    p = np.random.random(1).squeeze()
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
    if np.random.rand() < epsilon :
        return np.random.randint(n_outputs)
    return action

######## 경험 리플레이 생성 ########
#### 정의 ####
# 전이 정보 <s, a, s', r, done>이 저장

#### 변수 할당 ####
buffer_len = 20000
exp_buffer = deque(maxlen=buffer_len)
#### 함수 생성 ####
def sample_memories(batch_size) :
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    # np.random.permutation(n) : n 길이만큼의 순열을 랜덤으로 생성
    mem = np.array(exp_buffer)[perm_batch]

    return mem[:,0], mem[:,1], mem[:,2], mem[:,3], mem[:,4] # <s, a, s', r, done>

######## 히이퍼 파라미터 할당 ########
num_episodes = 800
batch_size = 48
input_shape = (None, 88, 80, 1)
learning_rate = 0.001
X_shape = (None, 88, 80, 1)
discount_factor = 0.97

global_step = 0
copy_steps = 100
steps_train = 4
start_steps = 2000
logdir = 'logs'

######## 플레이스 홀더 ########
X = tf.compat.v1.placeholder(tf.float32, shape=X_shape)
in_training_mode = tf.compat.v1.placeholder(tf.bool)

mainQ, mainQ_ouputs = q_network(X, 'mainQ')
targetQ, targetQ_ouputs = q_network(X, 'targetQ')

X_action = tf.compat.v1.placeholder(tf.int32, shape=(None,))
Q_action = tf.reduce_sum(targetQ_ouputs * tf.one_hot(X_action, n_outputs), axis=-1, keepdims=True) # 원소 합

# var dict에 저장된 key와 값을 받아 main이름과 targetQ 값을 복사
copy_op = [tf.compat.v1.assign(main_name, targetQ[var_name]) for var_name, main_name in mainQ.items()]
copy_target_to_main = tf.group(*copy_op)

y = tf.compat.v1.placeholder(tf.float32, shape=(None,1))

loss = tf.reduce_mean(tf.square(y - Q_action))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

######## 텐서보드 시각화 ########
loss_summary = tf.compat.v1.summary.scalar('Loss', loss)
merge_summary = tf.compat.v1.summary.merge_all()
file_writer = tf.compat.v1.summary.FileWriter(logdir, tf.get_default_graph())

######## 모델 실행 ########
init = tf.global_variables_initializer()
with tf.Session() as sess :
    init.run()

    for i in range(num_episodes) :
        print('-----------------')
        print("# of Episodes : ",i)
        done = False
        obs = env.reset()
        epoch = 0
        episodic_reward = 0
        actions_counter = Counter()
        episodic_loss = []

        while not done :
            #env.render()

            obs = preprocess_observation(obs)
            actions = mainQ_ouputs.eval(feed_dict={X:[obs], in_training_mode:False})

            action = np.argmax(actions, axis=-1)
            actions_counter[str(action)] += 1

            action = epsilon_greedy(action, global_step)

            next_obs, reward, done, _ = env.step(action)

            exp_buffer.append([obs, action, preprocess_observation(next_obs), reward, done])

            if global_step % steps_train == 0 and global_step > start_steps :
                o_obs, o_act, o_next_obs, o_rew, o_done = sample_memories(batch_size)

                o_obs = [x for x in o_obs]

                o_next_obs = [x for x in o_next_obs]

                next_act = mainQ_ouputs.eval(feed_dict={X:o_next_obs, in_training_mode:False})

                y_batch = o_rew + discount_factor * np.max(next_act, axis=-1) * (1-o_done)

                mrg_summary = merge_summary.eval(feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:False})

                file_writer.add_summary(mrg_summary, global_step)

                train_loss, _ = sess.run([loss, training_op], feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:True})
                episodic_loss.append(train_loss)

            if (global_step+1) % copy_steps == 0 and global_step > start_steps :
                copy_target_to_main.run()
            obs = next_obs
            epoch += 1
            global_step += 1
            episodic_reward += reward
        print('Epoch : ', epoch)
        print('Reward : ', episodic_reward)
        print('-----------------')
        print(' ')