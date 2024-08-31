import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

'''
solution.csv의 path에 대한 distance 계산 및 시각화 코드
현재까지 찾은 best : 25.0250

'''

def load_path_from_csv(filename='final_path.csv'):
    df = pd.read_csv(filename, header=None)
    path = df.iloc[:, 0].tolist()
    return path

def get_distance_matrix(x, num_cities=998):
    x = torch.tensor(x)
    x1, x2 = x[:,0:1], x[:,1:2]
    d1 = x1 - (x1.T).repeat(num_cities,1)
    d2 = x2 - (x2.T).repeat(num_cities,1)
    distance_matrix = (d1**2 + d2**2)**0.5
    return distance_matrix.numpy()

def calculate_path_cost(path, distance_matrix):
    cost = 0
    for i in range(len(path) - 1):
        cost += distance_matrix[path[i], path[i + 1]]
    cost += distance_matrix[path[-1], path[0]]
    return cost

def plot_path(coords, path, distance):
    plt.figure(figsize=(10, 8))
    for i in range(len(path) - 1):
        x_coords = [coords[path[i], 0], coords[path[i + 1], 0]]
        y_coords = [coords[path[i], 1], coords[path[i + 1], 1]]
        plt.plot(x_coords, y_coords, 'r-')
    x_coords = [coords[path[-1], 0], coords[path[0], 0]]
    y_coords = [coords[path[-1], 1], coords[path[0], 1]]
    plt.plot(x_coords, y_coords, 'r-')
    plt.plot(coords[:, 0], coords[:, 1], 'bo')
    plt.plot(coords[path[0], 0], coords[path[0], 1], 'go', markersize=10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Path Visualization / Distance: {distance:.4f}')
    plt.grid(True)
    plt.show()


path = load_path_from_csv('solution_01.csv')
coords = np.array(pd.read_csv('2024_AI_TSP.csv', header=None))
W = get_distance_matrix(coords)

distance = calculate_path_cost(path, W)
print("Total path cost:", distance)

plot_path(coords, path, distance)
