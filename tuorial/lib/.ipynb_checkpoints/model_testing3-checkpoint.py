import torch
from torch import nn
import gymnasium as gym
import numpy as np

class LR_block(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.linear = nn.Linear(shape[0], shape[1])
        self.relu = nn.ReLU()
        # 유전 알고리즘의 학습은 역전파를 통한 학습과 다르므로 배치 정규화와 드롭아웃은 사용해봐야 의미가 없다

    def forward(self, x):
        output = self.relu(self.linear(x))

        return output

class Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.body = nn.Sequential(*[LR_block(shape) for shape in layers[:-1]])
        self.tail_top = nn.Linear(layers[-1][0], layers[-1][1])
        self.tail_bottom = nn.LogSoftmax(dim=0)

    def forward(self, x):
        output = self.body(x)
        output = self.tail_bottom(self.tail_top(output))

        return output

def unpack_params(params, layers):
    # 만약 어떤 은닉층이 노드가 10개이고 크기가 4인 1차원 텐서가 입력이라면
    # 이 은닉층의 가중치 텐서는 10 X 4 형태의 텐서가 된다
    # params는 모델을 이루는 모든 계층들의 학습 가능한 매개변수들을 1차원 텐서 형태로 만들어 놓은 것이다(flatten)
    unpacked_params = []
    start = 0
    for l in layers:
        division = np.prod(l)
        stop = start + division + l[1]
        layer_vec = params[start:stop] # 각 계층의 매개변수 벡터를 추출한다
        unpacked_params.extend([layer_vec[0:division].reshape((l[1], l[0])), layer_vec[division:]]) # 각 계층에 해당하는 매개변수 벡터로부터 다시 가중치 벡터와 편향 벡터를 추출한다
        start = stop

    return unpacked_params

def reconstruct_state_dict(state_dict, parameters_set):
    for i, key in enumerate(state_dict):
        state_dict[key].copy_(parameters_set[i]) # 모델의 매개변수 텐서를 개체의 매개변수 텐서로 모두 덮어씌운다

    return state_dict

def max_gene(pop):
    max_fit = 0
    max_idx = 0
    for i, x in enumerate(pop):
        if x["fitness"] > max_fit:
            max_fit = x["fitness"]
            max_idx = i

    print(max_fit)
    return max_idx

def worker(pop, layers):
    ''' 학습 후 시험(test) '''
    max_idx = max_gene(pop)
    model = Model(layers)
    model.load_state_dict(reconstruct_state_dict(model.state_dict(), unpack_params(pop[max_idx]["params"], layers))) # 개체를 토대로 모델을 구성한다
    
    env = gym.make("CartPole-v1", render_mode="human") # 카트폴 환경 불러오기
    cur_state = torch.from_numpy(env.reset()[0]).float() # 환경 초기화 및 초기 상태 반환
    
    for i in range(100):
        with torch.no_grad():
            logits = model(cur_state)
        action = torch.distributions.categorical.Categorical(logits=logits).sample() # 옆의 코드는 카테고리컬 분포 말고 다항 분포로도 구현할 수 있다
                                                                                     # logit에 근거한 확률분포를 바탕으로 2개의 행동 중 하나를 뽑는다
        next_state, _, done, _, _ = env.step(action.item())
        print(done)
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