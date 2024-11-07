import multiprocessing as mp
import numpy as np
import os

print(f"__name__ : {__name__}")

def square(x):
    return np.square(x)

# 아래의 if문을 사용하지 않으면 무한 오류에 빠지게 된다
if __name__ == "__main__":
    x = np.arange(64) # 0 ~ 63까지의 정수 생성
    print(x, type(x), x.dtype)
    print("Current Host System cpu core number :", mp.cpu_count()) # 현재 호스트 시스템의 cpu(코어) 개수를 반환한다
                                                               # 그러나 이것이 현재 프로세스가 사용 가능한 cpu(코어)의 개수를 의미하지는 않는다
    pool = mp.Pool(8) # 8개의 프로세스를 가지는 프로세스 풀을 생성한다
    squared = pool.map(square, [x[8*i:8*i+8] for i in range(8)]) # 총 64개의 숫자를 8개씩 분할하여 각 프로세스에 배정한다
    print(squared)