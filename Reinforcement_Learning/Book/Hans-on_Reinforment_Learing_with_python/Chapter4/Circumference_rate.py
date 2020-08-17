import numpy as np
import math
import random
import matplotlib.pyplot as plt


######## 함수생성 ########

# 랜덤으로 점 생성
def generate_points(size) :
    x = random.random()*size
    y = random.random()*size
    return(x,y)

# 원 안에 점이 있는지 확인(return : True or False)
def is_in_circle(point, size) :
    return math.sqrt(point[0] ** 2 + point[1] ** 2) <= size

# 원주율 계산
def compute_pi(points_inside_circle, points_inside_square) :
    return 4 * (points_inside_circle / points_inside_square)



######## 변수 초기화 및 할당 ########

# 사각형 크기 설 및 호 생성
square_size = 1
arc = np.linspace(0, np.pi/2, 100)

# 변수 초기화
points_inside_circle = 0
points_inside_square = 0

# 샘플 개수 설정
sample_size = 1000

######## 점 생성 및 원 안에 있는 점의 개수 계산 ########
points_x = []
points_y = []

for i in range(sample_size) :
    point = generate_points(square_size)
    points_x.append(point[0])
    points_y.append(point[1])
    points_inside_square += 1

    if is_in_circle(point, square_size) :
        points_inside_circle +=1


######## 그래프 그리기 ########
plt.axes().set_aspect('equal')
plt.plot(1*np.cos(arc), 1*np.sin(arc))
plt.plot(points_x, points_y,'.', color='#ff0000')
print("Approximate value of pi is {} ".format(compute_pi(points_inside_circle, points_inside_square)))
plt.show()