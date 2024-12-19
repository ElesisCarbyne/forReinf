import torch
import gymnasium as gym
import numpy as np

def unpack_params(params, layers=[(25, 4), (10, 25), (2, 10)]):
    # 만약 어떤 은닉층이 노드가 10개이고 크기가 4인 1차원 텐서가 입력이라면
    # 이 은닉층의 가중치 텐서는 10 X 4 형태의 텐서가 된다
    # params는 모델을 이루는 모든 계층들의 학습 가능한 매개변수들을 1차원 텐서 형태로 만들어 놓은 것이다(flatten)
    unpacked_params = []
    e = 0
    for i, l in enumerate(layers):
        s, e = e, e + np.prod(l) # np.prod()는 주어진 축(axis) 상의 배열 원소들에 대한 곱을 수행한다
        weights = params[s:e].view(l) # 첫 번째 은닉층부터 시작하여 각 계층의 가중치 텐서를 params로부터 추출한다
        s, e = e, e+l[0]
        bias = params[s:e] # 첫 번째 은닉층부터 시작하여 각 계층의 편향 텐서를 params로부터 추출한다
        unpacked_params.extend([weights, bias])

    return unpacked_params

def model(x, unpacked_params):
    l1, b1, l2, b2, l3, b3 = unpacked_params
    y = torch.nn.functional.linear(x, l1, b1) # torch.nn.functional.linear()에는 입력과 가중치 텐서, 편향 텐서가 인자로 전달된다
    y = torch.relu(y)
    y = torch.nn.functional.linear(y, l2, b2)
    y = torch.relu(y)
    y = torch.nn.functional.linear(y, l3, b3)
    y = torch.log_softmax(y, dim=0)

    return y

def max_gene(pop):
    max_fit = 0
    max_idx = 0
    for i, x in enumerate(pop):
        if x["fitness"] > max_fit:
            max_fit = x["fitness"]
            max_idx = i
    return max_idx

def worker(pop):
    ''' 학습 후 시험(test) '''
    max_idx = max_gene(pop)
    model_params = unpack_params(pop[max_idx]["params"])
    
    env = gym.make("CartPole-v1", render_mode="human") # 카트폴 환경 불러오기
    cur_state = torch.from_numpy(env.reset()[0]).float() # 환경 초기화 및 초기 상태 반환
    
    for i in range(100):
        with torch.no_grad():
            logits = model(cur_state, model_params)
        action = torch.distributions.categorical.Categorical(logits=logits).sample() # 옆의 코드는 카테고리컬 분포 말고 다항 분포로도 구현할 수 있다
                                                                                     # logit에 근거한 확률분포를 바탕으로 2개의 행동 중 하나를 뽑는다
        next_state, _, done, _, _ = env.step(action.item())
        if done:
            print("Lost: rechead end of game")
            return
        cur_state = torch.from_numpy(next_state).float()
        
        env.render() # 에이전트가 현재 보고 있는 것의 시각화를 위해 환경을 렌더링 한다
                     # 이전에는 render() 메서드가 인자를 받았지만, 지금은 이 인자를 make() 메서드가 받도록 수정되었다
    print("Model did not failed on game!!!")

if __name__ == "__main__":
    print("This module runs directly now...")
    print("Please use this module in sub-process...")