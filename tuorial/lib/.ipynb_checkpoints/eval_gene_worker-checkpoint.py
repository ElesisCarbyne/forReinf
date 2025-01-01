''' 개체 평가 worker 프로세스 '''
import gymnasium as gym
import numpy as np
import torch
from torch import nn

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

def unpack_params(params_vec, layers):
    # 만약 어떤 은닉층이 노드가 10개이고 크기가 4인 1차원 텐서가 입력이라면
    # 이 은닉층의 가중치 텐서는 10 X 4 형태의 텐서가 된다
    # params는 모델을 이루는 모든 계층들의 학습 가능한 매개변수들을 1차원 텐서 형태로 만들어 놓은 것이다(flatten)
    unpacked_params = []
    start = 0
    for l in layers:
        division = np.prod(l)
        stop = start + division + l[1]
        layer_vec = params_vec[start:stop] # 각 계층의 매개변수 벡터를 추출한다
        unpacked_params.extend([layer_vec[0:division].reshape((l[1], l[0])), layer_vec[division:]]) # 각 계층에 해당하는 매개변수 벡터로부터 다시 가중치 벡터와 편향 벡터를 추출한다
        start = stop

    return unpacked_params

def reconstruct_state_dict(state_dict, parameters_set):
    for i, key in enumerate(state_dict):
        state_dict[key].copy_(parameters_set[i]) # 모델의 매개변수 텐서를 개체의 매개변수 텐서로 모두 덮어씌운다

    return state_dict

def evaluate(batch, layers, params_set, fitness_set):
    model = Model(layers)
    state_dict = model.state_dict()

    done = False
    score = 0
    env = gym.make("CartPole-v1")
    
    for idx in range(*batch):
        ''' 개체를 모델로 구현 '''
        model.load_state_dict(reconstruct_state_dict(state_dict, unpack_params(params_set[idx], layers)))

        ''' 개체 평가 '''
        cur_state = torch.from_numpy(env.reset()[0]).float()
        while not done: # done = True이면 반복문이 종료된다
            with torch.no_grad():
                logits = model(cur_state) # logits아닌 probs로 적혀있었던 것이 유전 알고리즘의 테스트에서 의도대로 작동하지 않았던 원인이었다
                                          # probs로 적어놨기 때문에 아래의 카테고리컬 분포 계산이 잘못 계산되었던 것이다
            action = torch.distributions.categorical.Categorical(logits=logits).sample() # logit을 바탕으로 카테고리컬 분포(즉, 확률분포)를 구성한 후, 그에 따라 행동 1개를 선택한다
            next_state, _, done, _, _ = env.step(action.item())
            cur_state = torch.from_numpy(next_state).float()
            score += 1 # 에피소드의 길이를 각 개체에 대한 적합도 점수로 간주한다
        
        fitness_set[idx] = score
        score = 0
        done = False
    print("here!!!")