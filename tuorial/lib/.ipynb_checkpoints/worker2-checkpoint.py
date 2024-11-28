import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.multiprocessing as mp
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

def worker(proc_num, counter, params, model_save_path):
    model = ActorCritic()
    model.share_memory() # share_memory() 메서드는 이를 호출한 텐서를 shared_memory로 이동시킨다
                         # 여기서는 shared_memory에 모델의(여기서는 ActorCritic()) 매개변수를 저장하여,
                         # 서로의 모델을 훈련시키려는 각 프로세스가 동일한 모델 매개변수를 공유하도록 한다
    processes = []
    '''
    <참고>
    https://a-researcher.tistory.com/24
    
    mp.Lock()은 non-recursive lock 객체를 생성하는데, 이를 통해 프로세스들이 공유 객체를 사용함에 있어 서로 간섭하지 못하도록 lock을 통해 우선 선점 후 사용하도록 한다
    Lock에는 아래와 같이 2종류가 존재한다
    - non-recursive lock : 단 한번의 lock만 획득하는 것(프로세스 간 또는 스레드 간 lock이 이에 해당한다)
    - recursive lock : lock을 획득한 상태에서 다시 lock을 획득하는 것
        - 일반적으로 lock을 두 번 호출하면 무한 대기 상태에 빠지게 되지만, recursive lock은 lock을 재귀적으로 획득할 수 있도록 허용한다(lock의 획득을 수반하는 함수를 재귀적으로 호출할 때 lock의 획득도 재귀적으로 할 수 있도록 허용한다)
        - 이렇게 재귀적으로 획득한 lock은 우선 순위의 의미가 생기며, 가장 마지막에 호출된 lock이 가장 높은 우선 순위를 갖는다
        - lock을 재귀적으로 획득하는 동안은 첫 lock을 통해 선점한 공유 자원에 대하여 계속 소유권을 가지고 있게 되며, lock을 획득한 횟수만큼 해제해 주어야 lock이 완전히 해제된다
        - recursive lock을 해석하면 "A라는 함수가 앞서 lock을 획득했는데, 나도 그 A라는 함수니까 해당 공유 자원에 접근할 수 있어. 그런데 내가 앞선 A함수보다 용건이 급해서 그 공유 자원을 먼저 사용할게(lock)." 이라고 할 수 있다
    '''
    lock = mp.Lock() # mp.Value()에 증감 연산과 같이 읽기-쓰기 작업을 동시에 수반하는 연산을 수행할 경우 lock을 먼저 획득해야 연산 결과에 대한 일관성이 유지된다

    start = t.time()
    for i in range(proc_num):
        p = mp.Process(target=learner, args=(i, model, counter, params, lock)) # args는 프로세스에게 할당할 작업의 인자를 의미한다
        p.start() # 프로세스를 실행한다
        processes.append(p)
    
    for p in processes:
        p.join() # join() 메서드를 호출한 (하위)프로세스가 작업을 완료하고 종료될 때까지 대기한다(block)
                 # join() 메서드를 사용하지 않으면 자식 프로세스는 유휴상태(idle)에 들어가고 종료되지 않아(부모 프로세스는 종료된다) 좀비 프로세스가 되어 손수 kill해줘야만 소멸하게 된다
                 # 즉, join() 메서드가 하는 일은 부모 프로세스가 자식 프로세스보다 먼저 종료되지 못하도록 막는다
                 # join() 메서드를 호출하면 하위 프로세스의 작업이 모두 완료된 후, 주 프로세스의 나머지 작업이 진행된다
    print(f"sub-sub-process cluster running time: {(t.time() - start) / 60:.4f} min")
    print("counter.value in sub-process: ", counter.value)
    
    torch.save(model.state_dict(), model_save_path) # 학습된 모델의 매개변수를 저장한다
    for i, p in enumerate(processes):
        print(f"sub-sub-process 0{i}'s exitcode: '{p.exitcode}") # 공유 객체에 저장된 값을 출력한다
                                                        # .exitcode는 자식 프로세스의 종료 코드(exit code)이다
                                                        # 자식 프로세스가 아직 종료되지 않았다면 "None"을 반환하고,
                                                        # 정상적으로 종료되었다면 "0"을 반환한다

    for p in processes:
        p.terminate() # 프로세스를 종료한다
                      # 부모 프로세스는 terminate() 메서드를 사용하지 않아도 자동으로 종료된다(하지만 자식 프로세스는 자동으로 종료되지 않는다)
        p.close() # 프로세스 객체를 해체하여 그것과 관련된 모든 자원들을 회수한다

def learner(t, worker_model, counter, params, lock):
    worker_env = gym.make("CartPole-v1") # 환경 불러오기
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())
    # ep_len_list = []

    for i in range(params["epochs"]):
        state_values, logprobs, rewards, G = run_episode_N(worker_env, worker_model, params["N_steps"])
        actor_loss, critic_loss, ep_len = update_params_N(worker_opt, state_values, logprobs, rewards, G)
        # ep_len_list.append(ep_len)
        with lock:
            counter.value = counter.value + 1
            print(counter.value)

    # print(ep_len_list)

def run_episode_N(worker_env, worker_model, N_steps=10):
    cur_state = torch.from_numpy(worker_env.reset()[0]).float()
    state_values, logprobs, rewards = [], [], []
    done = False
    j = 0
    G = torch.tensor([0]) # 반환값 계산에 부트스트래핑을 적용하기 위한 N 단계 상태의 상태 가치 함수 값을 저장한다
    
    while(j < N_steps and done == False):
        j += 1
        policy, state_value = worker_model(cur_state)
        state_values.append(state_value)
        action_dist = torch.distributions.categorical.Categorical(logits=policy) # 카테고리컬 분포는 시행 횟수 n이 1인 다항분포와 동일한 분포이다
                                                                                 # 여기서의 역할은 주어진 로짓을 확률분포로 변환하여 이 확률분포를 토대로 표본을 추출할 수 있도록 하는 것이다
        action = action_dist.sample()
        logprob_ = policy[action] # logprob_는 스칼라 텐서이다
        logprobs.append(logprob_)
        next_state, _, done, _, _ = worker_env.step(action.numpy())
        cur_state = torch.from_numpy(next_state).float()
        if done:
            reward = -10 # N 단계 이전에 게임이 종료되었다면 에피소드 내의 전체 상태에 대한 반환값을 직접 계산할 수 있으므로,
                         # 이 에피소드에 대해서는 반환값 계산에 G(상태 가치 함수값이자 반환값)를 사용하지 않는다는 의미에서 0값을 넣는다(그냥 위에서 초기화된 G를 반환하면 된다)
        else:
            reward = 1
            G = state_value.detach() # 게임이 종료되지는 않았지만 에피소드가 N 단계까지 진행되었다면 N 단계의 상태에 대한 상태 가치 함수 값을 저장한다
                                     # N 단계 상태에 대한 반환값은 N+1 단계 상태의 반환값을 구할 수 없으므로 계산할 수 없지만,
                                     # N-1 단계 상태의 반환값은 그 상태에서 받은 보상값과 (다음 상태인)N 단계 상태에 대한 상태 가치 함수값(N 단계 상태의 반환값과 같은 것으로 간주된다)으로 계산할 수 있기 때문이다
                                     # detach()를 해주어야 하는 이유는 이후에 G가 actor_loss 계산에 사용되는 Returns을 구하기 위해서 사용되는데 G는 비평자의 계산 그래프와 연결되어있다
                                     # 만약 이를 그대로 사용하게 되면 actor_loss로 행위자와 더불어 비평자까지 같이 학습을 하게되므로 이를 방지하기 위해 G에 detach()를 해주어야 한다
                                     # 즉, actor_loss가 비평자를 베제하고 행위자만을 학습시킬 수 있도록 하기 위함이다
        rewards.append(reward)

        # if j == 1:
        #     print(f"about state_value: {state_value}, {state_value.size()}, {type(state_value)}") # 크키가 1인 1차원 텐서이다
        #     print(f"about policy: {policy}, {policy.size()}, {type(policy)}") # 크기가 2인 1차원 텐서이다
        #     print(f"about action: {action}, {action.size()}, {type(action)}") # 스칼라 텐서이다
        
    return state_values, logprobs, rewards, G

def update_params_N(worker_opt, state_values, logprobs, rewards, G, clc=0.1, gamma=0.95):
    rewards = torch.tensor(rewards).flip(dims=(0,))
    logprobs = torch.stack(logprobs).flip(dims=(0,)) # torch.stack 대신 torch.tensor를 사용해도 된다
    state_values = torch.stack(state_values).flip(dims=(0,)).view(-1) # torch.stack 대신 torch.tensor를 사용해도 된다
    
    Returns = [] # 반환값 저장, G가 비평자의 계산 그래프와 분리되었으므로 Returns 역시 비평자의 계산 그래프와 분리된다
    ret_ = rewards[0] if G == 0 else G # 마지막 상태의 반환값을 계산한다
                                         # G == 0 : 에피소드가 N 단계 상태에 도달하기 전에 종료되었을 경우, G != 0 : 에피소드가 N 단계 상태에 도달하여 종료되었을 경우
    Returns.append(ret_)
    for r in range(rewards.shape[0] - 1): # 마지막 상태를 제외한 보상의 개수만큼 반복을 수행한다
        ret_ = rewards[r+1] + gamma * ret_  # 에피소드의 마지막 타임 스텝을 T라 했을 때, T-1 타임 스텝부터 반환값을 계산한다
        Returns.append(ret_)

    Returns = torch.stack(Returns).view(-1).float() # 텐서가 원소인 리스트를 torch.tensor를 통해 텐서로 변환하면 오류가 발생하는데, torch.stack을 사용하면 오류없이 텐서로 변환할 수 있다
    Returns = F.normalize(Returns, dim=0) # 반환값들에 대해 정규화를 수행하여 [-1.0, 1.0] 구간의 값으로 변환한다
                                          # 이것때문에 비평자의 출력에 tanh를 적용한 것이다
    actor_loss = -1 * logprobs * (Returns - state_values.detach()) # 크기가 n인 1차원 텐서이다
    critic_loss = torch.pow(state_values - Returns, 2) # 크기가 n인 1차원 텐서이다
    loss = actor_loss.sum() + clc * critic_loss.sum() # 행위자가 비평자보다 더 빨리 학습하도록 하기 위해 clc=0.1을 곱한다
                                                      # 비평자의 전체 손실 중 일부로만 역전파를 수행하여 비평자의 학습을 지연시킨다
    
    # 역전파 수행
    worker_opt.zero_grad()
    loss.backward()
    worker_opt.step()

    return actor_loss, critic_loss, len(rewards)