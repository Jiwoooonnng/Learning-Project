import gym
import numpy as np

# Generate Env
env = gym.make('FrozenLake-v0')

def compute_value_function(policy, gamma=1.0) :
    # Initialize value table
    value_table = np.zeros(env.nS)
    while(1) :
        updated_value_table = np.copy(value_table)

        for state in range(env.nS) :
            action = policy[state]
            temp_value = 0
            for next_sr in env.P[state][action]:
                # calculate value per each state and action
                trans_prob, next_state, reward_prob, _ = next_sr
                temp_value += trans_prob * (reward_prob + gamma * updated_value_table[next_state])
            # sum value per each state
            value_table[state] = temp_value
        threshold = 1e-10
        if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold) :
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

def policy_iteration(env,gamma = 1.0) :
    random_policy = np.zeros(env.observation_space.n)
    no_of_iterations = 200000
    for i in range(no_of_iterations) :
        # calculate value table
        new_value_function = compute_value_function(random_policy, gamma)
        # calculate policy
        new_policy = extract_policy(new_value_function, gamma)
        if( np.all(random_policy == new_policy)) :
            print('Policy-Iteration converged at step %d' %(i+1))
            break
        # allocate policy
        random_policy = new_policy

    return new_policy

print(policy_iteration(env))
env.render()
