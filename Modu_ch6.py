#%% ch.6 퍼셉트론

# AND gate
def AND(x1, x2) :
    if x1 == 0 or x2 == 0 :
        return 0
    else :
        return 1
    
# OR gate
def OR(x1, x2) :
    if x1 == 0 and x2 == 0 :
        return 0
    else :
        return 1

for x in [(0,0), (1,0), (0,1), (1,1)] :
    y1 = AND(x[0],x[1])
    print("입력 값 : " + str(x) + " 출력 값 : " + str(y1))
    y2 = OR(x[0],x[1])
    print("입력 값 : " + str(x) + " 출력 값 : " + str(y2))