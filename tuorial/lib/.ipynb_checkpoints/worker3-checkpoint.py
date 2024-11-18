import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.multiprocessing as mp
import numpy as np
import gymnasium as gym
import time as t

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

def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1") # 환경 불러오기
    try:
        worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())
    except:
        print("Error occur(B)...")
        return

    for i in range(params["epochs"]):
        state_values, logprobs, rewards = run_episode(worker_env, worker_model)
        actor_loss, critic_loss, ep_len = update_params(worker_opt, state_values, logprobs, rewards)
        counter.value = counter.value + 1

def run_episode(worker_env, worker_model):
    cur_state = torch.from_numpy(worker_env.reset()[0]).float()
    state_values, logprobs, rewards = [], [], []
    done = False # 에피소드 종료 여부

    while (done == False):
        policy, state_value = worker_model(cur_state)
        state_values.append(state_value)
        logits = policy.view(-1) # 1차원 텐서로 변환한다
        action_dist = torch.distributions.categorical.Categorical(logits=logits) # 카테고리컬 분포는 시행 횟수 n이 1인 다항분포와 동일한 분포이다
                                                                    # 여기서의 역할은 주어진 로짓을 확률분포로 변환하여 이 확률분포를 토대로 표본을 추출할 수 있도록 하는 것이다
        action = action_dist.sample()
        logprob_ = logits[action]
        logprobs.append(logprob_)
        next_state, _, done, _, _ = worker_env.step(action.numpy())
        cur_state = torch.from_numpy(next_state).float()
        if done:
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
        rewards.append(reward)

    return state_values, logprobs, rewards

def update_params(worker_opt, state_values, logprobs, rewards, clc=0.1, gamma=0.95):
    rewards = torch.tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1) # torch.stack 대신 torch.tensor를 사용해도 된다
    state_values = torch.stack(state_values).flip(dims=(0,)).view(-1) # torch.stack 대신 torch.tensor를 사용해도 된다
    Returns = [] # 반환값 저장
    ret_ = torch.tensor([0])
    for r in range(rewards.shape[0]): # 보상의 개수만큼 반복을 수행한다
        ret_ = rewards[r] + gamma * ret_ # 에피소드의 마지막 타임 스텝부터 반환값을 계산한다
        Returns.append(ret_)

    Returns = torch.stack(Returns).view(-1) # 텐서가 원소인 리스트를 torch.tensor를 통해 텐서로 변환하면 오류가 발생하는데, torch.stack을 사용하면 오류없이 텐서로 변환할 수 있다
    Returns = F.normalize(Returns, dim=0) # 반환값들에 대해 정규화를 수행하여 [-1.0, 1.0] 구간의 값으로 변환한다
                                          # 이것때문에 비평자의 출력에 tanh를 적용한 것이다
    actor_loss = -1 * logprobs * (Returns - state_values.detach())
    critic_loss = torch.pow(state_values - Returns, 2)
    loss = actor_loss.sum() + clc * critic_loss.sum() # 행위자가 비평자보다 더 빨리 학습하도록 하기 위해 clc=0.1을 곱한다
                                                      # 비평자의 전체 손실 중 일부로만 역전파를 수행하여 비평자의 학습을 지연시킨다
    # 역전파 수행
    worker_opt.zero_grad()
    loss.backward()
    worker_opt.step()

    return actor_loss, critic_loss, len(rewards)

if __name__ == "__main__":
    ''' 몬테카를로 방식 분산 이익 행위자-비평자 학습 '''
    MasterNode = ActorCritic()
    MasterNode.share_memory() # share_memory() 메서드는 이를 호출한 텐서를 shared_memory로 이동시킨다
                               # 여기서는 shared_memory에 모델의(여기서는 ActorCritic()) 매개변수를 저장하여,
                               # 서로의 모델을 훈련시키려는 각 프로세스가 동일한 모델 매개변수를 공유하도록 한다
    processes = []
    params = {
        "epochs":1000,
        "n_workers":7
    }
    
    counter = mp.Value("i", 0)

    start = t.time()
    for i in range(params["n_workers"]):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params)) # args는 프로세스에게 할당할 작업의 인자를 의미한다
        p.start() # 프로세스가 실제로 생성된다
        processes.append(p)
    
    for p in processes:
        p.join() # 프로세스가 종료될 때 까지 block시킨다
                 # join() 메서드를 사용하지 않으면 자식 프로세스는 유휴상태(idle)에 들어가고 종료되지 않아(부모 프로세스는 종료된다) 좀비 프로세스가 되어 손수 kill해줘야만 소멸하게 된다
                 # 즉, join() 메서드가 하는 일은 부모 프로세스가 자식 프로세스보다 먼저 종료되지 못하도록 막는다
    
    for p in processes:
        p.terminate() # 프로세스를 종료한다
                      # 부모 프로세스는 terminate() 메서드를 사용하지 않아도 자동으로 종료된다(하지만 자식 프로세스는 자동으로 종료되지 않는다)
    end = t.time()
    print(f"total time : {(end - start) / 60:.6f} min")
    
    print(counter.value, processes[1].exitcode) # 공유 객체에 저장된 값을 출력한다
                                                # .exitcode는 자식 프로세스의 종료 코드(exit code)이다
                                                # 자식 프로세스가 아직 종료되지 않았다면 "None"을 반환하고,
                                                # 정상적으로 종료되었다면 "0"을 반환한다