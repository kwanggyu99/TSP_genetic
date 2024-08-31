import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import pickle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Device 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Hyperparameters 설정
EPOCHES = 500                       # 학습 에포크 수
INIT_LR = 0.0001                    # 초기 학습률
PATH_COST_SCALING_FACTOR = 0.001    # path_cost 스케일링 값
PCA_COMPONENTS = 100                # PCA 차원 수
LOG_DIR = 'graph/DNN_Q_Learning_1'




# FC Layer 모델 정의
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x




# q-table 로드 및 정규화
def load_and_normalize_q_table(file_path):
    with open(file_path, 'rb') as f:
        q_table = pickle.load(f)

    min_value = q_table.min()   # 정규화
    if min_value < 0:
        q_table = q_table + abs(min_value)
    min_value = q_table.min()
    max_value = q_table.max()
    q_table = q_table / (max_value - min_value)

    return q_table

# 거리 행렬
def get_distance_matrix(x, num_cities=998):
    x = torch.tensor(x)
    x1, x2 = x[:,0:1], x[:,1:2]
    d1 = x1 - (x1.T).repeat(num_cities,1)
    d2 = x2 - (x2.T).repeat(num_cities,1)
    distance_matrix = (d1**2 + d2**2)**0.5   # Euclidean Distance
    return distance_matrix.numpy()

# q-table 통해 최적 path 구성
def extract_optimal_path(q_table, start_city=0):
    num_cities = q_table.shape[0]
    current_city = start_city
    path = [current_city]
    visited = set(path)
    
    for _ in range(num_cities - 1):
        q_values = q_table[current_city].clone()
        q_values[list(visited)] = -float('inf')  # 이미 방문한 도시는 무시
        next_city = torch.argmax(q_values).item()
        path.append(next_city)
        visited.add(next_city)
        current_city = next_city

    return path

# distance 계산
def calculate_path_cost(path, distance_matrix):
    cost = 0
    for i in range(len(path) - 1):
        cost += distance_matrix[path[i], path[i + 1]]
    cost += distance_matrix[path[-1], path[0]]  # 돌아오는 경로 비용
    return cost

# path 출력
def print_partial_path(path, num_cities_to_print=8):
    print(f'{path[:num_cities_to_print]} ... {path[-num_cities_to_print:]}')

# path 저장
def save_path_to_csv(path, filename='solution_01.csv'):
    df = pd.DataFrame(path)
    df.to_csv(filename, index=False, header=False)





# TSP data load
coords = np.array(pd.read_csv('2024_AI_TSP.csv', header=None))

# 거리 행렬 생성
W_np = get_distance_matrix(coords)
print('998x998 거리 행렬:')
print(W_np)

# q-table 로드 및 스케일링
q_table_path = 'q_table.pkl'
q_table = load_and_normalize_q_table(q_table_path)
print('\n998x998 Q-table:')
print(q_table)

# 학습 전 path 출력
initial_path = extract_optimal_path(torch.tensor(q_table, dtype=torch.float32))
print('\nInitial path before training:')
print_partial_path(initial_path)
initial_distance = calculate_path_cost(initial_path, W_np)

# PCA로 거리 행렬 차원 축소
pca = PCA(n_components=PCA_COMPONENTS)
W_pca = pca.fit_transform(W_np)

# 모델 초기화
input_dim = PCA_COMPONENTS  # PCA로 축소된 크기
output_dim = 998 * 998  # 동일한 크기의 출력
model = QNetwork(input_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
writer = SummaryWriter(log_dir=LOG_DIR)



print('\n#########################################################################\n')

# 학습 과정
for epoch in tqdm(range(EPOCHES)):
    model.train()
    
    # PCA로 축소된 거리 행렬을 사용하여 각 도시 간의 특성 벡터 생성
    input_distance_matrix = torch.tensor(W_pca, dtype=torch.float32).to(device)
    
    # 도시 간의 평균 특성 벡터로 하나의 입력 벡터 생성
    input_vector = input_distance_matrix.mean(dim=0)
    
    # 모델 예측
    output = model(input_vector)
    
    # 출력 q-table을 다시 2차원 형태로 변환
    output_q_table = output.view(998, 998)
    
    # target q-table을 현재 가지고 있는 Q-테이블로 설정
    target_q_table = torch.tensor(q_table, dtype=torch.float32).to(device)

    # 예측 q-table로 최적 경로 비용 계산
    optimal_path = extract_optimal_path(output_q_table.detach())
    path_cost = calculate_path_cost(optimal_path, W_np)

    # Loss 계산 (MSE + 경로 비용 가중치 계산)
    loss = nn.functional.mse_loss(output_q_table, target_q_table)

    # 네트워크 학습
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # TensorBoard에 기록
    writer.add_scalar('distance/epoch', path_cost, epoch)
    writer.add_scalar('loss/epoch', loss.item(), epoch)

    # 학습 과정 출력
    print(f'\n\nEpoch [{epoch + 1}/{EPOCHES}], Path Cost: {path_cost:.4f}, Loss: {loss.item():.4f}')

    
final_output = model(input_vector)
final_q_table = final_output.view(998, 998).detach().cpu().numpy()
final_path = extract_optimal_path(torch.tensor(final_q_table, dtype=torch.float32))
print('\nFinal path after training:')
print_partial_path(final_path)

# # solution path 저장
# save_path_to_csv(final_path)

writer.close()
