#%% ch.4 경사하강법
#%% 단순선형회귀 경사하강법
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 공부시간 x와 성적 y의 리스트
data = [[2,81], [4,93], [6,91], [8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# 그래프
plt.figure(figsize=(8,5))
plt.scatter(x,y)            # scatter(산점도) plot
plt.show()

# 리스트 x,y를 넘파이 배열로 변환(인덱스를 줘서 하나씩 불러와서 계산이 가능하도록)
x_data = np.array(x)
y_data = np.array(y)

# 기울기a, 절편b initialization
a = 0
b = 0

# 학습률(Learning Rate)
lr = 0.03

# 몇 번 반복
epochs = 2001

# 경사하강법
for i in range(epochs) :        # 에포크 수만큼 반복
    y_pred = a * x_data + b     # y예측값 공식
    error = y_data - y_pred     # 오차 공식
    # differential of Cost func in a
    a_diff = -(2/len(x_data))*sum(x_data*(error))
    # differential of Cost func in b
    b_diff = -(2/len(x_data)*sum(error))
    
    a = a - lr * a_diff         # a update
    b = b - lr * b_diff         # b updata
    
    if i % 100 == 0 :            # 100번 반복할 때마다 update현황 출력
        print('epoch=%.f, 기울기=%.04f, 절편=%.04f' % (i,a,b))
        
# 그래프 그리기
y_pred = a * x_data + b
plt.scatter(x,y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()

#%% 다중선형회귀 경사하강법
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d    # 3D 그래프 그리는 라이브러리

# 공부시간x, 성적y 리스트
data = [[2,0,81], [4,4,93], [6,2,91], [8,3,97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

# 그래프로 확인
ax = plt.axes(projection='3d')      # 그래프 유형 정하기
ax.set_xlabel('study_hours')
ax.set_ylabel('private_class')
ax.set_zlabel('Score')
ax.dist = 11
ax.scatter(x1,x2,y)
plt.show()

# x,y를 넘파이 배열로 변환
x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

# 기울기a, 절편b initialization
a1 = 0
a2 = 0
b = 0

# Learning rate
lr = 0.02

# epoch
epochos = 2001

# 경사하강법
for i in range(epochs) :
    y_pred = a1*x1_data + a2*x2_data + b   # y구하는 공식
    error = y_data - y_pred             # 오차 공식
    a1_diff = -(2/len(x1_data))*sum(x1_data*(error))    # differential of Cost func in a1
    a2_diff = -(2/len(x2_data))*sum(x2_data*(error))    # differential of Cost func in a2
    b_diff = -(2/len(x1_data))*sum(y_data-y_pred)       # differential of Cost func in b
    a1 = a1 - lr*a1_diff    # a1 update
    a2 = a2 - lr*a2_diff    # a2 update
    b = b - lr*b_diff       # b update
    
    if i % 100 == 0 :
        print('epoch = %.f, 기울기1 = %.04f, 기울기2 = %.04f, 절편 = %.04f' % (i,a1,a2,b))
