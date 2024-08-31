import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import csv
import math
import pickle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm




# DQN 신경망 정의
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4*output_dim)
        self.fc2 = nn.Linear(4*output_dim, 2*output_dim)
        self.fc2_1 = nn.Linear(2*output_dim, output_dim)
        self.fc3 = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2_1(x))
        return self.fc3(x)

# DQN 학습 함수 정의 (MC 샘플링 사용)
def train_dqn(dqn, memory, batch_size, log_interval=1000, log_episode=0):
    if len(memory) < batch_size:
        return

    # batch 정보 불러오기
    transitions = memory.sample(batch_size)
    batch_state, batch_action, batch_return = zip(*transitions)
    
    batch_state = np.array(batch_state)
    batch_state = torch.tensor(batch_state, dtype=torch.float32).to(device)
    batch_action = torch.tensor(batch_action, dtype=torch.int64).to(device)
    batch_return = torch.tensor(batch_return, dtype=torch.float32).to(device)

    # loss 계산
    current_q_values = dqn(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
    expected_q_values = batch_return
    loss = criterion(current_q_values, expected_q_values)
    
    # 학습
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # # batch 로깅
    # if log_episode % log_interval == 0:
    #     print(f"\nLogging for Episode {log_episode}:")
    #     print(f"Batch Action (first 5): {batch_action[:5].cpu().numpy()}")
    #     print(f"Batch Return(Target) (first 5): {batch_return[:5].cpu().numpy()}")
    #     print(f"Current Q Values (first 5): {current_q_values[:5].cpu().detach().numpy()}")

    return loss.item()




# 경험 메모리 (MC 샘플링 추가) 순환 큐 형식
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, transitions):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transitions
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    
    
    

# action 선택 함수
def select_action(state, visited):
    global epsilon
    if np.random.rand() < epsilon:  # random 선택
        return np.random.choice(list(set(range(num_cities)) - visited))
    else:   # best 선택
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = dqn(state_tensor).cpu().numpy()  # 텐서를 넘파이 배열로 변환
            for i in visited:
                q_values[i] = -float('inf')
            return np.argmax(q_values)

# 현재 네트워크 모델에서 최적 경로 추출
def find_optimal_path(dqn, num_cities):
    state = np.zeros(state_dim)
    current_city = 0
    visited_cities = [current_city]
    state[current_city] = 1
    state[num_cities + current_city] = 1

    for _ in range(num_cities - 1):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = dqn(state_tensor).cpu().numpy()  # 텐서를 넘파이 배열로 변환
            for i in visited_cities:
                q_values[i] = -float('inf')
            action = np.argmax(q_values)
        next_city = action
        visited_cities.append(next_city)
        state[next_city] = 1
        state[num_cities + next_city] = 1
        current_city = next_city

    return visited_cities

# distance 계산
def calculate_path_cost(path, distance_matrix):
    cost = 0
    for i in range(len(path) - 1):
        cost += distance_matrix[path[i], path[i + 1]]
    cost += distance_matrix[path[-1], path[0]]  # 돌아오는 경로 비용
    return cost

# 유클리드 거리 계산 함수
def euclidean_distance(city1, city2):
    x1, y1 = float(city1[0]), float(city1[1])
    x2, y2 = float(city2[0]), float(city2[1])
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)





# 하이퍼파라미터 설정
global gamma, epsilon
writer = SummaryWriter(log_dir='graph/Deep_Q_Network_1')
num_episodes = 3000                 # 에피소드 수
batch_size = 128                    # 배치 크기
gamma = 0.95                        # target 계산 시 next state에 대한 discount factor
epsilon_start = 1.0
epsilon = epsilon_start             # 학습 시 다음 action 선택 기준
epsilon_end = 0.01
epsilon_decay = 0.995
max_memory = 10000                  # 메모리 크기
learning_rate_start = 0.0001
learning_rate = learning_rate_start # 학습률
learning_rate_end = 0.00001
learning_rate_decay = 0.999
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# 도시 정보 불러오기
cities = []
with open(r'2024_AI_TSP.csv', mode='r', newline='') as tsp:
    reader = csv.reader(tsp)
    for row in reader:
        cities.append(row)
num_cities = len(cities)
distances = np.zeros((num_cities, num_cities))

for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            city1, city2 = cities[i], cities[j]
            distance = euclidean_distance(city1, city2)
            distances[i, j] = distance


# state : 도시 수와 방문한 도시의 집합 / action : 현재 state에서의 q-value
state_dim = num_cities * 2
action_dim = num_cities


dqn = DQN(state_dim, action_dim).to(device)
memory = ReplayMemory(max_memory)
optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
criterion = nn.MSELoss()





# DQN 학습 루프
for episode in tqdm(range(num_episodes)):
    print(f"\nEpisode {episode + 1} out of {num_episodes} -------------------------------------------------- |")
    epsilon = max(epsilon_end, epsilon_decay * epsilon)
    learning_rate = max(learning_rate_end, learning_rate_decay * learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    state = np.zeros(state_dim)
    current_city = np.random.randint(0, num_cities)
    visited_cities = set([current_city])
    state[current_city] = 1  # 현재 도시 표시
    state[num_cities + current_city] = 1  # 방문한 도시 표시

    # 다음 action 선택 및 reward 계산
    episode_memory = []
    episode_reward = 0
    while len(visited_cities) < num_cities:
        action = select_action(state, visited_cities)
        next_city = action
        reward = -distances[current_city, next_city]
        next_state = state.copy()
        next_state[next_city] = 1
        next_state[num_cities + next_city] = 1
        episode_memory.append((state, action, reward, next_state))
        state = next_state
        visited_cities.add(next_city)
        current_city = next_city
        episode_reward += reward

    # 마지막에 시작 도시로 돌아가기
    visited_cities_list = list(visited_cities)
    reward = -distances[current_city, visited_cities_list[0]]
    episode_memory.append((state, visited_cities_list[0], reward, state))
    episode_reward += reward

    # target 계산 및 메모리에 추가
    for i in range(len(episode_memory)):
        state, action, reward, next_state = episode_memory[i]
        
        with torch.no_grad():
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
            next_q_values = dqn(next_state_tensor)
            max_next_q_value = next_q_values.max().item()
        
        target = reward + gamma * max_next_q_value
        memory.push((state, action, target))
    

    # 네트워크 학습
    loss = train_dqn(dqn, memory, batch_size)
    optimal_path = find_optimal_path(dqn, num_cities)
    total_distance = calculate_path_cost(optimal_path, distances)
    
    # tensorboard에 기록
    writer.add_scalar('Loss/episode', loss, episode)
    writer.add_scalar('Distance/episode', total_distance, episode)
    writer.add_scalar('Reward/episode', episode_reward, episode)
    
    print(f"\nLoss: {loss:.6f}, Leraning Rate: {learning_rate:.6f}, Rewards: {episode_reward:.2f}")
    print(f"Path Cost: {total_distance:.4f}")
    if (episode+1) % 10 == 0:
        print(f"{episode+1}회 학습 결과 최적의 경로: {optimal_path[:7]} ... {optimal_path[-7:]}")
    print()

writer.close()
print("DQN 학습 완료")
