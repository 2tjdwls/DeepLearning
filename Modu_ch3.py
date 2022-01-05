모두의 딥러닝
#%% <ch.3> 선형회귀
* 임의의 직선을 그어 이에 대한 평균 제곱 오차를 구하고, 이 값을 최소화하는 a와 b를 찾는 과정
#%%최소제곱법
import numpy as np

# x값,y값
x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

# x,y평균
mx = np.mean(x)
my = np.mean(y)
print('x의 평균 : ',mx)
print('y의 평균 : ',my)

# 기울기 공식의 분모
divisor = sum([(i - mx)**2 for i in x])

# 기울기 공식의 분자
def top(x, mx, y, my) :
    d = 0
    for i in range(len(x)) :
        d += (x[i] - mx) * (y[i] - my)
    return d
dividend = top(x, mx, y, my)
print('분모 : ',divisor)
print('분자 : ',dividend)

# 기울기, y절편
a = dividend / divisor
b = my - (mx * a)

print('기울기 a = ',a)
print('y절편 = ',b)

#%%평균제곱오차(mean square error, MSE) : 오차 평가 알고리즘
import numpy as np

# 기울기 a, y절편 b
fake_a_b = [3, 76]

# x,y의 데이터 값
data = [[2,81], [4,93], [6,91], [8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# y = ax + b에 a와 b값을 대입해서 결과를 출력하는 함수 정의
def predict(x) :
    return fake_a_b[0]*x + fake_a_b[1]

# MSE 함수
def mse(y, y_hat) : # y_hat : y의 예측값
    return ((y-y_hat)**2).mean()

# MSE 함수를 각 y값에 대입하여 최종 값을 구하는 함수
def mse_val(y, predict_result) :
    return mse(np.array(y), np.array(predict_result))

# 예측 값이 들어갈 빈 리스트
predict_result = []

# 모든 x값을 한번씩 대입
for i in range(len(x)) : 
    # predict_result 리스트
    predict_result.append(predict(x[i]))
    print('공부한 시간 = %.f, 실제 점수 = %.f, 예측 점수 = %.f' % (x[i], y[i], predict(x[i])))
    
# 최종 MSE
print('MSE : ' + str(mse_val(y,predict_result)))

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