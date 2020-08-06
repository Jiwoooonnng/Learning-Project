import gym
import numpy as np

# Generate Env
env = gym.make('FrozenLake-v0')
print(env.observation_space.n)
print(env.action_space.n)

def value_iteration(env, gamma = 1.0) :
    # Initialize Value table
    value_table = np.zeros(env.observation_space.n)
    no_of_iterations = 100000

    # Update Value
    for i in range(no_of_iterations):
        updated_value_table = np.copy(value_table)

        # Generate Q_value(action, value)
        for state in range(env.observation_space.n) :
            Q_value = []

            for action in range(env.action_space.n) :
                next_states_rewards = []

                for next_sr in env.P[state][action] :
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))

                Q_value.append(np.sum(next_states_rewards))
                # Pick up the max Q_value and Update it as value of a state
                value_table[state] = max(Q_value)

        # Finish Trig
        threshold = 1e-20
        if(np.sum(np.fabs(updated_value_table - value_table)) <= threshold) :
            print('Value-iteration converged at iteration# %d.' %(i+1))
            break

    return value_table


def extract_policy(value_table, gamma = 1.0) :
    # Initialize Policy
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n) :

        # Initialize Q_table
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n) :
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr

                # Update Q_table
                Q_table[action] += trans_prob * (reward_prob + gamma * value_table[next_state])
        policy[state] = np.argmax(Q_table)
    return policy

value_table = value_iteration(env)
optimal_policy = extract_policy(value_table)
print(optimal_policy)
