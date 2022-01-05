#%% ch.5 로지스틱 회귀
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 공부시간x, 합격여부y 리스트
data = [[2,0], [4,0], [6,0], [8,1], [10,1], [12,1], [14,1]]

x_data = [i[0] for i in data]   # 공부한 시간 데이터
y_data = [i[1] for i in data]   # 합격 여부

# 그래프
plt.scatter(x_data,y_data)
plt.xlim(0, 15)
plt.ylim(-0.1, 1.1)

# 기울기a,절편b initialization
a = 0
b = 0

# Learning Rate
lr = 0.05

# sigmoid func 정의
def sigmoid(x) :
    return 1/(1 + np.e ** (-x))     # sigmoid function fomula

# 에포크
epochs = 2001

# 경사하강법
for i in range(epochs) :
    for x_data, y_data in data :
        a_diff = x_data * (sigmoid(a*x_data + b) - y_data)  # partial diff in a
        b_diff = sigmoid(a*x_data + b) - y_data             # partial diff in b
        a = a - lr * a_diff     # a update
        b = b - lr * b_diff     # b update
        if i % 1000 == 0 :
            print('epoch = %.f, 기울기 = %.04f, 절편 = %.04f' % (i,a,b))
            
 # 각 횟수마다 그래프
plt.scatter(x_data, y_data)
plt.xlim(0,15)
plt.ylim(-0.1, 1.1)
x_range = (np.arange(0, 15, 0.1))
plt.plot(np.arange(0, 15, 0.1), np.array([sigmoid(a * x + b) for x in x_range]))
plt.show()