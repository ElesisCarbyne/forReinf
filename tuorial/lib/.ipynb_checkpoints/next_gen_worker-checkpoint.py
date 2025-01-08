''' 다음 개체군 생성 worker 프로세스 '''
import numpy as np
import torch
import random

def recombine(x1, x2):
    split_pt = np.random.randint(x1.shape[0]) # pt = point, 부모 개체의 교차점을 무작위로 추출한다
    # 자손 개체 생성
    child1 = torch.cat((x1[0:split_pt], x2[split_pt:]), dim=0) # 부모 텐서와는 완전히 별개인 새로운 텐서가 생성된다
    child2 = torch.cat((x2[0:split_pt], x1[split_pt:]), dim=0) # 부모 텐서와는 완전히 별개인 새로운 텐서가 생성된다

    return child1, child2

def mutate(x, rate=0.01):
    num_to_change = int(rate*x.shape[0]) # 자손 개체의 매개변수들 중 변이시킬 매개변수 비율을 결정한다
    idx = random.sample(range(0, x.shape[0]), k=num_to_change) # 결정된 비율만큼의 매개변수 인덱스를 반환한다
    x[idx] = torch.randn(num_to_change) / 10.0 # 선택된 매개변수들을 모두 무작위 실수 값으로 변경한다
                                               # 10.0으로 굳이 나누어 준 것은 모든 무작위 실수 값들이 0.xxx의 형태로 표현되도록 만들어 주기 위함이다
    return x

def next_generation(tour_quota, mut_rate, tournament_size, params_set, fitness_set, event_count):
    lp = params_set.shape[0] # 개체군의 크기
    temp = []

    ''' 자손 개체 생성 '''
    print(f"tour_quota : {tour_quota}")
    for i in range(*tour_quota):
        # 토너먼트 생성 및 부모 개체 선정
        tournament = random.sample(range(0, lp), k=int(tournament_size*lp)) # 토너먼트에 참여할 개체를 추출한다
        tournament = np.array([[idx, fitness_set[idx]] for idx in tournament]) # 토너먼트를 재구성한다
        tournament = tournament[tournament[:, 1].argsort()] # 토너먼트에 속한 개체들을 적합도 점수를 기준으로 내림차순으로 정렬한 결과가 반환된다
        p1, p2 = params_set[tournament[-1][0]], params_set[tournament[-2][0]]

        # 자식 개체 생성
        offsprings = recombine(p1, p2)
        temp.extend([mutate(offsprings[0], mut_rate), mutate(offsprings[1], mut_rate)])

    ''' 현재 개체군에 생성된 자손 개체를 덮어씌우기 '''
    # 모든 worker 프로세스가 할당받은 토너먼트로부터 자손 개체의 생성을 완료할 때까지 대기(그렇지 않으면 다른 프로세스가 현세대 개체군으로 자손 개체를 생성할 수 없다)
    event_count += 1
    while True:
        if event_count == 6 : break
    print(f"event_count : {event_count}")
    
    # 1번 프로세스가 30개의 토너먼트(토너먼트 범위 [0:30])를 처리했다면 60개의 자손 개체가 생성된다
    # 그리고 이 60개의 자손 개체는 현세대 개체군의 인덱스 범위 [0:60]에 대입한다
    # 마찬가지의 논리로 2번 프로세스로부터(토너먼트 범위 [30:60]) 생성된 60개의 자손 개체는 현세대 개체군의 인덱스 범위 [60:120]에 대입한다
    # 위와 같은 과정을 모든 프로세스에서 수행하여 기존의 개체군을 새로운 개체군으로 갱신하도록 하는 것이 아래 인덱스 슬라이싱의 의미이다
    params_set[2*tour_quota[0]:2*tour_quota[1]] = torch.stack(temp, dim=0)[:]