from torch import optim
import torch
from torch.nn import functional as F
import gymnasium as gym

def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1") # 환경 불러오기
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())

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
        action_dist = torch.ditributions.categorical.Categorical(logtis=logits) # 카테고리컬 분포는 시행 횟수 n이 1인 다항분포와 동일한 분포이다
                                                                    # 여기서의 역할은 주어진 로짓을 확률분포로 변환하여 이 확률분포를 토대로 표본을 추출할 수 있도록 하는 것이다
        action = action_dist.sample()
        logprob_ = logits[action]
        logprobs.append(logprob_)
        next_state, _, done, _, _ = worker_env.step(action.numpy())
        cur_state = torh.from_numpy(next_state).float()
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