import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.multiprocessing as mp
import gymnasium as gym
import time as t
import os

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.normalize(x, dim=0) # F.normalize()에서 도출되는 결과의 범위는 [-1.0, 1.0]이다
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y), dim=0) # 음의 로그 확률 값을 모델 단에서 미리 계산한다(정확히는 그냥 로그 확률이다)
        c = F.relu(self.l3(y.detach())) # 여기서 계산 그래프가 분리된다
        critic = torch.tanh(self.critic_lin1(c)) # 가치 함수 값을 tanh을 사용하여 [-1.0, 1.0] 구간의 값으로 변환시켜준 것은
                                                 # 이익 계산에서의 Returns를 정규화하여 [-1.0, 1.0] 구간의 값으로 변환하기 때문이다
        return actor, critic

def worker(model_save_path):
    ''' 학습 후 시험(test) '''
    model = ActorCritic()
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    env = gym.make("CartPole-v1", render_mode="human") # 카트폴 환경 불러오기
    
    for i in range(100):
        cur_state = torch.from_numpy(env.reset()[0]).float() # 환경 초기화 및 초기 상태 반환
        logits, value = model(cur_state)
        action_dist = torch.distributions.categorical.Categorical(logits=logits) # 옆의 코드는 카테고리컬 분포 말고 다항 분포로도 구현할 수 있다
        action = action_dist.sample() # logit에 근거한 확률분포를 바탕으로 2개의 행동 중 하나를 뽑는다
        next_state, reward, done, _, _ = env.step(action.numpy())
        if done:
            print("Lost: rechead end of game")
            cur_state = torch.from_numpy(env.reset()[0]).float()
        env.render() # 에이전트가 현재 보고 있는 것의 시각화를 위해 환경을 렌더링 한다
                     # 이전에는 render() 메서드가 인자를 받았지만, 지금은 이 인자를 make() 메서드가 받도록 수정되었다
    print("model did not failed on game!!!")

if __name__ == "__main__":
    ''' 학습 후 시험(test) '''
    model_save_path = "D:\\for_study\\workspace\\for_reinf\\tuorial\\parameters\\test01.p"
    
    model = ActorCritic()
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    env = gym.make("CartPole-v1", render_mode="human") # 카트폴 환경 불러오기
    
    for i in range(100):
        cur_state = torch.from_numpy(env.reset()[0]).float() # 환경 초기화 및 초기 상태 반환
        logits, value = model(cur_state)
        action_dist = torch.distributions.categorical.Categorical(logits=logits) # 옆의 코드는 카테고리컬 분포 말고 다항 분포로도 구현할 수 있다
        action = action_dist.sample() # logit에 근거한 확률분포를 바탕으로 2개의 행동 중 하나를 뽑는다
        next_state, reward, done, _, _ = env.step(action.numpy())
        if done:
            print("Lost: rechead end of game")
            cur_state = torch.from_numpy(env.reset()[0]).float()
        env.render() # 에이전트가 현재 보고 있는 것의 시각화를 위해 환경을 렌더링 한다
                     # 이전에는 render() 메서드가 인자를 받았지만, 지금은 이 인자를 make() 메서드가 받도록 수정되었다
    print("model did not failed on game!!!")