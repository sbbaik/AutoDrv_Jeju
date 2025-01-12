import random
import numpy as np

# 데이터셋과 주사위 눈 초기화
X = [5, 3, 5, 5, 2, 4, 1, 6, 3, 1]  # 데이터셋
eyes = [1, 2, 3, 4, 5, 6]  # 주사위 눈
p = np.zeros(6)  # 각 눈의 확률 초기화

# 생성 모델 학습 함수
def learn_generator(X, p):
    for i in range(len(X)): 
        p[X[i] - 1] += 1  # 각 눈의 빈도수 증가
    p /= len(X)  # 각 눈의 확률 계산
    return p  # 학습된 확률 반환

# 샘플 생성 함수
def generate():
    return random.choices(eyes, p)  # 학습된 확률로 샘플 생성

# 모델 학습
p = learn_generator(X, p)

# 학습된 분포 출력
print("Learned Probability Distribution:")
for i in range(len(p)):
    print(f"Eye {i+1}: {p[i]:.2f}")

# 생성 결과 출력
print("\nGenerated Samples:")
for _ in range(4):  # 네 번 생성
    print(generate())
