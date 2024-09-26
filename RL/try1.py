import numpy as np

class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, num_actions))
        
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state])
        
    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error
        
def simulate_environment(num_episodes, agent, num_states):
    for episode in range(num_episodes):
        state = 0  
        total_reward = 0
        
        while state != num_states - 1:
            action = agent.choose_action(state)
            next_state, reward = take_action(state, action, num_states)
            agent.update_q_table(state, action, reward, next_state)
            total_reward += reward
            state = next_state
        
        print("Episode {}: Total Reward = {}".format(episode + 1, total_reward))
        
def take_action(state, action, num_states): 
    if action == 0:  # Move right
        if state < num_states - 1:
            next_state = state + 1
        else:
            next_state = state
    elif action == 1:  # Move left
        if state > 0:
            next_state = state - 1
        else:
            next_state = state
    reward = 1 if next_state == num_states - 1 else -1
    return next_state, reward

if __name__ == "__main__":
    num_states = 5  
    num_actions = 2  
    agent = QLearningAgent(num_states, num_actions)
    num_episodes = 10
    simulate_environment(num_episodes, agent, num_states)

print("Q-table after {} episodes:".format(num_episodes))
print(agent.q_table)