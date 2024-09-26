import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import shutil
from collections import deque
import random
import pygame
import sys

# Define the grid world size
GRID_SIZE = 10
# Define the maximum path length threshold
MAX_PATH_LENGTH = 100
# Define the size of the experience replay buffer
REPLAY_BUFFER_SIZE = 10000
# Define the batch size for training
BATCH_SIZE = 32

# Define actions
ACTIONS = {
    0: 'UP',
    1: 'RIGHT',
    2: 'DOWN',
    3: 'LEFT'
}

# Q-network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# Deep Q-Network Agent
class DQNAgent:
    def __init__(self, input_size, output_size, lr=0.001, gamma=0.99, epsilon=0.1):
        self.q_network = QNetwork(input_size, output_size)
        self.target_network = QNetwork(input_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 4)  # Random action
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state).float())
                return torch.argmax(q_values).item()

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        # Sample a minibatch from the replay buffer
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        states = torch.tensor([exp[0] for exp in minibatch]).float()
        actions = torch.tensor([exp[1] for exp in minibatch]).long()
        rewards = torch.tensor([exp[2] for exp in minibatch]).float()
        next_states = torch.tensor([exp[3] for exp in minibatch]).float()
        dones = torch.tensor([exp[4] for exp in minibatch]).float()

        q_values = self.q_network(states)
        q_values_next = self.target_network(next_states)

        max_next_q_values = torch.max(q_values_next, dim=1)[0]
        targets = rewards + (1 - dones) * self.gamma * max_next_q_values

        q_values_selected = q_values[range(BATCH_SIZE), actions]

        loss = self.loss_fn(q_values_selected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def add_to_replay_buffer(self, experience):
        self.replay_buffer.append(experience)

# Grid Environment
class GridWorld:
    def __init__(self, grid_size, initial_state, goal_state):
        self.grid_size = grid_size
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.current_state = initial_state
        self.obstacles = set()

    def reset(self):
        self.current_state = self.initial_state
        return self.current_state

    def add_obstacle(self, obstacle):
        if obstacle != self.initial_state and obstacle != self.goal_state:
            self.obstacles.add(obstacle)

    def step(self, action):
        if action == 0:  # UP
            next_state = (max(0, self.current_state[0] - 1), self.current_state[1])
        elif action == 1:  # RIGHT
            next_state = (self.current_state[0], min(self.grid_size - 1, self.current_state[1] + 1))
        elif action == 2:  # DOWN
            next_state = (min(self.grid_size - 1, self.current_state[0] + 1), self.current_state[1])
        elif action == 3:  # LEFT
            next_state = (self.current_state[0], max(0, self.current_state[1] - 1))

        if next_state in self.obstacles:
            next_state = self.current_state  # Don't move into an obstacle

        self.current_state = next_state
        done = (self.current_state == self.goal_state)
        reward = 1 if done else 0
        return self.current_state, reward, done

    def is_valid_state(self, state):
        return 0 <= state[0] < self.grid_size and 0 <= state[1] < self.grid_size and state not in self.obstacles

    def find_shortest_path(self):
        from queue import Queue

        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        queue = Queue()
        queue.put((self.initial_state, [self.initial_state]))
        visited = set()
        visited.add(self.initial_state)

        while not queue.empty():
            current_state, path = queue.get()
            if current_state == self.goal_state:
                return path
            for d in directions:
                next_state = (current_state[0] + d[0], current_state[1] + d[1])
                if self.is_valid_state(next_state) and next_state not in visited:
                    queue.put((next_state, path + [next_state]))
                    visited.add(next_state)
        return []

# Pygame visualization and user interaction
def run_pygame():
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption('Grid World')

    cell_size = 500 // GRID_SIZE
    initial_state = None
    goal_state = None
    optimal_path = []
    obstacles = set()

    agent = None
    env = None

    def draw_grid():
        for x in range(0, 500, cell_size):
            for y in range(0, 500, cell_size):
                rect = pygame.Rect(x, y, cell_size, cell_size)
                pygame.draw.rect(screen, (0, 0, 0), rect, 1)

    def draw_points():
        if initial_state is not None:
            pygame.draw.circle(screen, (0, 255, 0), (initial_state[1] * cell_size + cell_size // 2, initial_state[0] * cell_size + cell_size // 2), 15)
        if goal_state is not None:
            pygame.draw.circle(screen, (255, 0, 0), (goal_state[1] * cell_size + cell_size // 2, goal_state[0] * cell_size + cell_size // 2), 15)
        for obstacle in obstacles:
            pygame.draw.rect(screen, (0, 0, 0), (obstacle[1] * cell_size, obstacle[0] * cell_size, cell_size, cell_size))

    def draw_path(path):
        if path:
            for i in range(len(path) - 1):
                start_pos = (path[i][1] * cell_size + cell_size // 2, path[i][0] * cell_size + cell_size // 2)
                end_pos = (path[i + 1][1] * cell_size + cell_size // 2, path[i + 1][0] * cell_size + cell_size // 2)
                pygame.draw.line(screen, (0, 0, 255), start_pos, end_pos, 5)

    def reset_env():
        nonlocal agent, env, optimal_path
        if initial_state is not None and goal_state is not None:
            env = GridWorld(GRID_SIZE, initial_state, goal_state)
            agent = DQNAgent(2, 4)
            agent.update_target_network()
            optimal_path = []

    def handle_mouse_click(pos):
        nonlocal initial_state, goal_state, obstacles, env
        grid_x = pos[0] // cell_size
        grid_y = pos[1] // cell_size
        clicked_square = (grid_y, grid_x)
        if initial_state is None:
            initial_state = clicked_square
            reset_env()
        elif goal_state is None:
            goal_state = clicked_square
            reset_env()
        else:
            obstacles.add(clicked_square)
            if env is not None:
                env.add_obstacle(clicked_square)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:
                    handle_mouse_click(pygame.mouse.get_pos())
                elif pygame.mouse.get_pressed()[2]:  # Right click to start finding the path
                    if env is not None and initial_state is not None and goal_state is not None:
                        optimal_path = env.find_shortest_path()

        screen.fill((255, 255, 255))
        draw_grid()
        draw_points()
        draw_path(optimal_path)
        pygame.display.flip()

    pygame.quit()

run_pygame()
