import random
import pickle
from collections import defaultdict
from pokerSim import get_global_rewards
from tqdm import tqdm  # Ensure tqdm is installed and imported
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

# Initialize the Q-table with optimistic initial values
q_table = defaultdict(lambda: defaultdict(lambda: 10.0))  # Optimistic Q-values

# Initialize belief states for opponent strategies
belief_states = {
    "SB": {"aggressive": 0.3, "passive": 0.5, "balanced": 0.2},
    "BB": {"aggressive": 0.2, "passive": 0.6, "balanced": 0.2},
    "UTG": {"aggressive": 0.1, "passive": 0.7, "balanced": 0.2},
    "MP": {"aggressive": 0.2, "passive": 0.5, "balanced": 0.3},
    "CO": {"aggressive": 0.4, "passive": 0.3, "balanced": 0.3},
}

# Initialize action counters for logging
action_counters = defaultdict(lambda: defaultdict(int))

# Existing helper functions
def get_q_value(state, action):
    """Retrieve the Q-value for a state-action pair."""
    return q_table[state][action]

def update_q_value(state, action, value):
    """Update the Q-value for a state-action pair."""
    q_table[state][action] = value

def save_q_table(filename="q_table.pkl"):
    """Save the Q-table to a file as a regular dictionary."""
    with open(filename, "wb") as f:
        # Convert nested defaultdicts to regular dicts
        regular_q_table = {state: dict(actions) for state, actions in q_table.items()}
        pickle.dump(regular_q_table, f)

def load_q_table(filename="q_table.pkl"):
    """Load the Q-table from a file into a defaultdict."""
    global q_table
    try:
        with open(filename, "rb") as f:
            loaded_q_table = pickle.load(f)
            # Convert loaded regular dicts into defaultdicts
            q_table = defaultdict(lambda: defaultdict(lambda: 10.0), {
                state: defaultdict(lambda: 10.0, actions) for state, actions in loaded_q_table.items()
            })
        print("Loaded existing Q-table.")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        print(f"No valid Q-table found ({e}). Starting fresh.")
        q_table = defaultdict(lambda: defaultdict(lambda: 10.0))  # Reset Q-table with optimistic values

def update_belief_state(position, observed_action):
    """
    Update the belief state based on observed opponent action.
    """
    if position not in belief_states:
        return  # No belief state for this position

    strategies = belief_states[position]
    # Update based on observed_action
    if observed_action == "raise":
        strategies["aggressive"] += 0.1
    elif observed_action in ["call", "fold"]:
        strategies["passive"] += 0.1
    else:
        strategies["balanced"] += 0.1  # For other actions

    # Decrease other strategies
    for strategy in strategies:
        if (strategy == "aggressive" and observed_action != "raise") or \
           (strategy == "passive" and observed_action not in ["call", "fold"]):
            strategies[strategy] -= 0.05

    # Normalize probabilities to sum to 1
    total = sum(strategies.values())
    for strategy in strategies:
        strategies[strategy] = max(strategies[strategy] / total, 0.01)  # Avoid zero probabilities

def sample_opponent_strategy(position):
    """
    Sample an opponent strategy based on the belief state.
    """
    strategies = list(belief_states[position].keys())
    probabilities = list(belief_states[position].values())
    return random.choices(strategies, probabilities)[0]

def get_hand_strength(hand_key, global_rewards):
    """
    Calculate normalized hand strength between 0 and 1.
    """
    reward = global_rewards.get(hand_key, (0,))[0]
    # Normalize hand strength based on assumed range
    normalized_strength = (reward + 20) / 40
    return min(max(normalized_strength, 0), 1)

def calculate_reward(action, global_reward, pot_size, opponent_strategy, hand_strength):
    raise_penalty_factor = 0.2  # Reduced penalty factor for raising

    if action == "fold":
        if hand_strength < 0.3:
            return 1.0  # Reward for folding very weak hands
        elif hand_strength < 0.6:
            return 0.5  # Moderate reward for folding medium-weak hands
        else:
            return -0.5  # Penalty for folding strong hands

    elif action == "call":
        call_risk = 0.2 * pot_size if opponent_strategy == "aggressive" else 0.1 * pot_size
        return (global_reward - call_risk) / 50  # Balanced scaling

    elif action.startswith("raise"):
        raise_levels = {"raise_100": 100, "raise_200": 200, "raise_300": 300}
        bet_size = raise_levels.get(action, 100)
        raise_success_chance = 0.3 if opponent_strategy == "passive" else 0.2

        # Calculate reward with reduced penalty scaled by hand strength
        penalty = raise_penalty_factor * bet_size * (1 - hand_strength)
        reward = (global_reward + (bet_size * raise_success_chance)) - (0.2 * pot_size) - penalty

        # Apply bonuses based on hand strength and raise level appropriateness
        if bet_size == 100:
            if hand_strength >= 0.5:
                reward += 1  # Encourage raise_100 with medium hands
            else:
                reward -= 0.5  # Discourage raise_100 with weak hands
        elif bet_size == 200:
            if hand_strength >= 0.6:
                reward += 2  # Encourage raise_200 with slightly stronger hands
            else:
                reward -= 1  # Discourage raise_200 with weaker hands
        elif bet_size == 300:
            if hand_strength >= 0.7:
                reward += 3  # Encourage raise_300 with very strong hands
            else:
                reward -= 2  # Discourage raise_300 with weaker hands

        # Normalize the reward
        return reward / 50

    elif action == "check":
        return (global_reward * 0.6) / 50  # Balanced scaling for check

    return 0

def get_optimal_action(state):
    """
    Retrieve the optimal action based on the current Q-values for a given state.
    """
    actions = ["check", "call", "fold", "raise_100", "raise_200", "raise_300"]
    q_values = {action: get_q_value(state, action) for action in actions}
    return max(q_values, key=q_values.get, default="check")

def simulate_opponent_action(strategy):
    """
    Simulate an opponent action based on their strategy.
    """
    if strategy == "aggressive":
        return random.choices(["raise", "call", "fold"], weights=[0.6, 0.3, 0.1])[0]
    elif strategy == "passive":
        return random.choices(["fold", "call", "raise"], weights=[0.5, 0.4, 0.1])[0]
    else:  # balanced
        return random.choices(["raise", "call", "fold"], weights=[0.3, 0.4, 0.3])[0]

def evaluate_policy(global_rewards, num_trials=100):
    """
    Evaluate the current policy by running multiple trials and calculating the average reward.
    """
    total_reward = 0
    for _ in range(num_trials):
        # Initialize a random starting state
        hand_key = random.choice(list(global_rewards.keys()))
        position = random.choice(["SB", "BB", "UTG", "MP", "CO"])
        state = (hand_key, position)

        # Sample an opponent strategy based on belief state
        opponent_strategy = sample_opponent_strategy(position)

        # Get the optimal action based on current Q-values
        action = get_optimal_action(state)

        # Get the global reward for the current hand
        global_reward = global_rewards[hand_key][0]

        # Calculate a random pot size for this trial
        pot_size = random.randint(50, 500)

        # Calculate hand strength
        hand_strength = get_hand_strength(hand_key, global_rewards)

        # Calculate the reward based on action, context, and sampled opponent strategy
        reward = calculate_reward(action, global_reward, pot_size, opponent_strategy, hand_strength)

        # Update the total reward with the reward from this trial
        total_reward += reward

    # Calculate and return the average reward
    average_reward = total_reward / num_trials
    return average_reward

def evaluate_policy_random(global_rewards, num_trials=1000):
    """
    Evaluate the random policy by running multiple trials and calculating the average reward.
    """
    total_reward = 0
    for _ in range(num_trials):
        # Initialize a random starting state
        hand_key = random.choice(list(global_rewards.keys()))
        position = random.choice(["SB", "BB", "UTG", "MP", "CO"])
        state = (hand_key, position)

        # Sample an opponent strategy based on belief state
        opponent_strategy = sample_opponent_strategy(position)

        # Choose a random action
        action = random.choice(["check", "call", "fold", "raise_100", "raise_200", "raise_300"])

        # Get the global reward for the current hand
        global_reward = global_rewards[hand_key][0]

        # Calculate a random pot size for this trial
        pot_size = random.randint(50, 500)

        # Calculate hand strength
        hand_strength = get_hand_strength(hand_key, global_rewards)

        # Calculate the reward based on action, context, and sampled opponent strategy
        reward = calculate_reward(action, global_reward, pot_size, opponent_strategy, hand_strength)

        # Update the total reward with the reward from this trial
        total_reward += reward

    # Calculate and return the average reward
    average_reward = total_reward / num_trials
    return average_reward

def choose_action(state, epsilon):
    """
    Choose an action based on epsilon-greedy policy.
    """
    actions = ["check", "call", "fold", "raise_100", "raise_200", "raise_300"]
    if random.random() < epsilon:
        # Assign higher weights to raising actions to encourage their selection during exploration
        return random.choices(
            actions, 
            weights=[0.05, 0.05, 0.3, 0.2, 0.2, 0.2]
        )[0]
    else:
        return get_optimal_action(state)  # Exploit

def extract_q_values(test_hands, positions, actions):
    """
    Extract Q-values for specified hands and positions.

    Parameters:
        test_hands (list): List of hand strings (e.g., ["AA", "KK", "72o"]).
        positions (list): List of position strings (e.g., ["SB", "BB"]).
        actions (list): List of action strings (e.g., ["check", "call"]).

    Returns:
        pd.DataFrame: DataFrame containing Q-values with columns:
                      ['Hand', 'Position', 'Action', 'Q_Value']
    """
    data = []
    for hand in test_hands:
        for pos in positions:
            state = (hand, pos)
            for action in actions:
                q_val = get_q_value(state, action)
                data.append({
                    'Hand': hand,
                    'Position': pos,
                    'Action': action,
                    'Q_Value': q_val
                })
    df = pd.DataFrame(data)
    return df

def plot_heatmap(q_values_df):
    """
    Plot heatmaps of average Q-values per hand and action across positions.

    Parameters:
        q_values_df (pd.DataFrame): DataFrame containing Q-values.
    """
    # Calculate average Q-values per hand, action, and position
    pivot_table = q_values_df.pivot_table(
        index='Hand',
        columns=['Position', 'Action'],
        values='Q_Value',
        aggfunc='mean'
    )
    
    # Plot the heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='viridis')
    plt.title('Heatmap of Q-Values for Hands, Positions, and Actions')
    plt.ylabel('Hand')
    plt.xlabel('Position and Action')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_faceted_bar_charts(q_values_df):
    """
    Plot faceted bar charts of Q-values for each action across all states.

    Parameters:
        q_values_df (pd.DataFrame): DataFrame containing Q-values.
    """
    g = sns.FacetGrid(q_values_df, col="Action", col_wrap=3, height=5, sharex=False, sharey=False)
    g.map_dataframe(sns.barplot, x="Hand", y="Q_Value", hue="Position")
    g.add_legend()
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Q-Values for Each Action Across Hands and Positions')
    plt.show()

def plot_interactive_bar_charts(q_values_df):
    """
    Plot interactive bar charts of Q-values for each action across all states.

    Parameters:
        q_values_df (pd.DataFrame): DataFrame containing Q-values.
    """
    fig = px.bar(
        q_values_df,
        x='Hand',
        y='Q_Value',
        color='Position',
        facet_col='Action',
        facet_col_wrap=3,
        title='Interactive Q-Values for Each Action Across Hands and Positions',
        labels={'Q_Value': 'Q-Value'},
        height=800
    )
    fig.update_layout(showlegend=True)
    fig.show()

def plot_boxplots(q_values_df):
    """
    Plot box plots of Q-values across different actions and hands.

    Parameters:
        q_values_df (pd.DataFrame): DataFrame containing Q-values.
    """
    plt.figure(figsize=(20, 10))
    sns.boxplot(x='Action', y='Q_Value', hue='Hand', data=q_values_df)
    plt.title('Distribution of Q-Values Across Actions and Hands')
    plt.xlabel('Action')
    plt.ylabel('Q-Value')
    plt.legend(title='Hand', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def q_learning_training(episodes=500000, alpha=0.2, gamma=0.9, 
                       epsilon_start=1.0, epsilon_end=0.05, 
                       epsilon_decay=0.9994, 
                       eval_interval=1000):
    """
    Train the Q-table using Q-learning with a progress bar.
    Collect performance data for both Q-learning and random policies.
    """
    global q_table
    epsilon = epsilon_start
    global_rewards = get_global_rewards()

    # Initialize tqdm progress bar
    training_rewards = []
    random_rewards = []
    epsilon_values = []
    episode_range = range(1, episodes + 1)
    
    for episode in tqdm(episode_range, desc="Training Progress"):
        # Initialize a random starting state
        hand_key = random.choice(list(global_rewards.keys()))
        position = random.choice(["SB", "BB", "UTG", "MP", "CO"])
        state = (hand_key, position)

        # Calculate hand strength
        hand_strength = get_hand_strength(hand_key, global_rewards)

        done = False
        steps = 0  # Track steps within an episode
        max_steps = 10  # Prevent infinite loops
        total_episode_reward = 0  # Track total reward for this episode

        while not done and steps < max_steps:
            steps += 1
            # Choose an action using epsilon-greedy policy
            action = choose_action(state, epsilon)

            # Increment action counter for the state
            action_counters[state][action] += 1

            # Sample an opponent strategy based on belief state
            opponent_strategy = sample_opponent_strategy(position)

            # Get the global reward for the current hand
            global_reward = global_rewards[hand_key][0]

            # Calculate a random pot size for this trial
            pot_size = random.randint(50, 500)

            # Calculate the reward based on action, context, and sampled opponent strategy
            reward = calculate_reward(action, global_reward, pot_size, opponent_strategy, hand_strength)

            # Update total episode reward
            total_episode_reward += reward

            # Simulate opponent action
            opponent_action = simulate_opponent_action(opponent_strategy)
            update_belief_state(position, opponent_action)

            # Define next state based on opponent's action
            if opponent_action in ["fold", "call"]:
                next_state = None  # Terminal state
                done = True
            else:
                # Transition to a new state, e.g., next round or similar
                next_hand_key = random.choice(list(global_rewards.keys()))
                next_position = random.choice(["SB", "BB", "UTG", "MP", "CO"])
                next_state = (next_hand_key, next_position)

                # Calculate next hand strength
                next_hand_strength = get_hand_strength(next_hand_key, global_rewards)

            # Get the maximum Q-value for the next state
            if next_state is None:
                max_future_q = 0
            else:
                actions_next = ["check", "call", "fold", "raise_100", "raise_200", "raise_300"]
                max_future_q = max([get_q_value(next_state, a) for a in actions_next], default=0)

            # Current Q-value
            current_q = get_q_value(state, action)

            # Q-learning update
            new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
            update_q_value(state, action, new_q)

            # Transition to next state
            state = next_state
            if next_state is not None:
                hand_strength = next_hand_strength

        # Decay epsilon
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay
            epsilon = max(epsilon, epsilon_end)
        
        # Record rewards at evaluation intervals
        if episode % eval_interval == 0:
            avg_q_reward = evaluate_policy(global_rewards, num_trials=1000)
            avg_random_reward = evaluate_policy_random(global_rewards, num_trials=1000)
            training_rewards.append(avg_q_reward)
            random_rewards.append(avg_random_reward)
            epsilon_values.append(epsilon)
            tqdm.write(f"Episode {episode}: Q-learning Avg Reward = {avg_q_reward:.4f}, Random Avg Reward = {avg_random_reward:.4f}, Epsilon = {epsilon:.4f}")

    return training_rewards, random_rewards, epsilon_values

def plot_performance(training_rewards, random_rewards, eval_interval=1000):
    """
    Plot the average rewards of Q-learning and Random policies over training episodes.
    """
    episodes = [i * eval_interval for i in range(1, len(training_rewards) + 1)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, training_rewards, label='Q-Learning Policy', color='blue')
    plt.plot(episodes, random_rewards, label='Random Policy', color='red', linestyle='--')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Performance Comparison: Q-Learning vs Random Policy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rewards_over_time(training_rewards, random_rewards, eval_interval=1000):
    """
    Plot cumulative rewards over time for both Q-learning and Random policies.
    """
    episodes = [i * eval_interval for i in range(1, len(training_rewards) + 1)]
    cumulative_q = []
    cumulative_random = []
    total_q = 0
    total_random = 0
    
    for q, r in zip(training_rewards, random_rewards):
        total_q += q
        total_random += r
        cumulative_q.append(total_q)
        cumulative_random.append(total_random)
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, cumulative_q, label='Q-Learning Cumulative Reward', color='green')
    plt.plot(episodes, cumulative_random, label='Random Cumulative Reward', color='orange', linestyle='--')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_epsilon_decay(epsilon_values, eval_interval=1000):
    """
    Plot the decay of epsilon over training episodes.
    """
    episodes = [i * eval_interval for i in range(1, len(epsilon_values) + 1)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, epsilon_values, label='Epsilon', color='purple')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon Value')
    plt.title('Epsilon Decay Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_q_values_distribution(states):
    """
    Plot the distribution of Q-values for specified states.
    """
    for state in states:
        actions = ["check", "call", "fold", "raise_100", "raise_200", "raise_300"]
        q_values = [get_q_value(state, action) for action in actions]
        
        plt.figure(figsize=(8, 4))
        plt.bar(actions, q_values, color='skyblue')
        plt.xlabel('Actions')
        plt.ylabel('Q-Values')
        plt.title(f'Q-Values Distribution for State: {state}')
        plt.grid(axis='y')
        plt.show()

# ... [All your existing helper functions] ...

# Load global rewards
global_rewards = get_global_rewards()

# Optional: Load existing Q-table if available
load_q_table()

# Train the Q-learning agent and collect performance data
print("Starting Q-learning training...")
training_rewards, random_rewards, epsilon_values = q_learning_training(episodes=100000, eval_interval=1000)  # Adjust the number of episodes as needed
print("Training completed.")

# Plot average rewards over time
plot_performance(training_rewards, random_rewards, eval_interval=1000)

# Plot cumulative rewards over time
plot_rewards_over_time(training_rewards, random_rewards, eval_interval=1000)

# Plot epsilon decay
plot_epsilon_decay(epsilon_values, eval_interval=1000)

# Test states after training
test_states = ["42o", "AA", "KK", "72o", "JTs", "AQs"]
positions = ["MP", "CO", "SB", "BB", "UTG"]

for hand in test_states:
    for pos in positions:
        state = (hand, pos)
        optimal_action = get_optimal_action(state)
        print(f"Optimal action for state {state}: {optimal_action}")

# After training, plot Q-values for specific states
specific_states = [("AA", "SB"), ("KK", "MP"), ("JTs", "CO"), ("AQs", "UTG")]
plot_q_values_distribution(specific_states)

# Additional Visualizations: Plot Q-values across all tested hands and positions
# Define the hands and positions you want to visualize
visualization_hands = ["42o", "AA", "KK", "72o", "JTs", "AQs"]  # Adjust as needed
visualization_positions = ["MP", "CO", "SB", "BB", "UTG"]  # Adjust as needed
visualization_actions = ["check", "call", "fold", "raise_100", "raise_200", "raise_300"]

# Extract Q-values for the specified hands and positions
q_values_df = extract_q_values(visualization_hands, visualization_positions, visualization_actions)

# Plot Heatmap
plot_heatmap(q_values_df)

# Plot Faceted Bar Charts
plot_faceted_bar_charts(q_values_df)

# Plot Interactive Bar Charts (Optional: Uncomment to use)
# plot_interactive_bar_charts(q_values_df)

# Plot Box Plots
plot_boxplots(q_values_df)
