import random
from pokerSim import get_global_rewards

q_table = {}

def get_q_value(state, action):
    return q_table.get(state, {}).get(action, 0.0)

def update_q_value(state, action, value):
    if state not in q_table:
        q_table[state] = {}
    q_table[state][action] = value


def q_learning(global_rewards, num_trials=10000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Perform Q-learning with a simplified state representation and prioritized folding for negative rewards.
    
    :param global_rewards: Global dictionary with hand keys and average rewards.
    :param num_trials: Number of Q-learning episodes to run.
    :param alpha: Learning rate.
    :param gamma: Discount factor.
    :param epsilon: Exploration rate.
    """
    for _ in range(num_trials):
        # Initialize a random starting state (simplified)
        hand_key = random.choice(list(global_rewards.keys()))
        position = random.choice(['SB', 'BB', 'UTG', 'MP', 'CO'])
        state = (hand_key, position)  # Simplified state representation

        # Get the global reward for the current hand
        global_reward = global_rewards[hand_key][0]

        # Choose an action using epsilon-greedy policy
        if global_reward < 0:
            # Bias towards fold for negative-reward hands
            if random.random() < epsilon:
                action = random.choice(['fold', 'check'])  # Limited exploration
            else:
                action = 'fold'  # Exploit the fold action
        else:
            # Normal epsilon-greedy policy for non-negative-reward hands
            if random.random() < epsilon:
                action = random.choice(['check', 'call', 'fold', 'raise'])
            else:
                q_values = {a: get_q_value(state, a) for a in ['check', 'call', 'fold', 'raise']}
                action = max(q_values, key=q_values.get, default='check')

        # Simulate the reward for the chosen action
        if action == 'fold':
            reward = -5  # Example penalty for folding
        elif action == 'call':
            reward = global_reward - 10  # Example current bet deduction
        elif action == 'raise':
            reward = (global_reward - 20) * 1.2  # Example additional raise penalty
        elif action == 'check':
            reward = global_reward * 0.8  # Example slight penalty for passive play

        # Transition to the next state (simplified, stays the same for preflop logic)
        next_state = state

        # Update Q-value
        max_next_q = max(get_q_value(next_state, a) for a in ['check', 'call', 'fold', 'raise'])
        current_q = get_q_value(state, action)
        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
        update_q_value(state, action, new_q)



def get_optimal_action(state):
    """
    Retrieve the optimal action for a given state based on the Q-table.
    :param state: Tuple (hand_key, position, pot_size, current_bet).
    :return: Optimal action as a string.
    """
    print(q_table)
    if state not in q_table:
        print("not in q")
    q_values = q_table.get(state, {})
    print(q_values)
    return max(q_values, key=q_values.get, default='check')

def print_sorted_q_table():
    """
    Print the Q-table with states sorted by actions alphabetically.
    """
    print("Q-table:")
    for state, actions in q_table.items():
        sorted_actions = dict(sorted(actions.items()))  # Sort actions alphabetically
        print(f"{state}: {sorted_actions}")




# Simulate Q-learning
global_rewards = get_global_rewards()

q_learning(global_rewards, num_trials=5000)
print_sorted_q_table()
# Test the optimal policy for a specific state
test_state = ('AA', 'CO')
optimal_action = get_optimal_action(test_state)
print(f"Optimal action for state {test_state}: {optimal_action}")
