import torch
import numpy as np
import torch.multiprocessing as mp

def square(x):
    return np.square(x)

def square2(i, x, queue):
    print("In process {}".format(i,))
    queue.put(np.square(x))

def square3(i, x, queue):
    temp = torch.pow(x, 2)
    print("In process {}: {}".format(i, temp))
    return queue.put(temp)

def square4(i, x, queue):
    print("In process {}: {}".format(i, x))
    return queue.put(x)

if __name__ == "__main__":
    processes = [] # 프로세스 풀 생성
    queue = mp.Queue() # 프로세스가 반환하는 결과물을 취합하기 위한 큐 생성
                       # mp.Queue를 사용하면 주 프로세스가 큐로부터 데이터를 읽어오기 전에 하위 프로세스가 종료되어 주 프로세스의 큐 읽기 작업에 오류가 발생할 수 있다
                       # "발생할 수 있다"라고 언급한 것 처럼 그렇지 않은 경우도 있긴하다
    # x = np.arange(64)
    x = torch.arange(1, 65)
    
    print("Current available Host System cpu core number :", mp.cpu_count())
    
    # mp.Pool()은 어떤 프로세스에게 어떤 작업을 배정할지를 자동으로 결정하는 반면,
    # mp.Process()는 어떤 프로세스에게 어떤 작업을 배정할지를 수동으로 지정한다
    # 프로세스 작업 수행 과정은 CMD에서 확인할 수 있다(.run() 메서드를 사용하면 직접 출력해 볼 수 있다)
    for i in range(8):
        start_index = 8 * i
        # multiprocessing에서 process는 Process 객체가 생성되고 이 객체의 start() 메서드가 호출되었을 때 소환된다(spawn)
        # Process 객체는 분리된 프로세스 내에서 실행중인 활동을 나타낸다
        proc = mp.Process(target=square3, args=(i, x[start_index:start_index+8], queue)) # 프로세스 번호, 작업물, 결과물 저장 큐를 넘겨준다
        # proc.run() # 프로세스의 활동을 출력한다
                     # 이 메서드를 사용하지 않으면 (쥬피터의 경우) CMD에 프로세스 활동이 출력된다
        proc.start() # 프로세스를 시작한다(프로세스가 실제로 생성된다)
        processes.append(proc) # 추후 프로세스들을 제어하기 위해 프로세스들을 추적한다


    results = []
    count = 0
    while count != 8:
        if not queue.empty(): # 큐가 비어있지 않을 경우, 큐 안의 결과물을 반환한다
            results.append(queue.get())
            count += 1
    for i in range(len(results)):
        print(results[i])
    
    for proc in processes:
        proc.join() # 프로세스가 종료될 때까지 기다린다
                    # 좀비 프로세스가 생성되는 것을 막기 위해 가급적이면 .join() 을 사용해서 프로세스를 종료시키는 것이 좋다
        
    # for proc in processes:
    #     proc.terminate() # 프로세스를 종료한다(자원 반환은 수행하지 않는다)
    #                      # 부모 프로세스를 종료하더라도 그 밑의 자식 프로세스들은 종료되지 않는다(이 경우 자식 프로세스들은 "기아"가 된다)
    
    # for proc in processes:
    #     proc.close() # 프로세스 객체를 해체하여 그것과 관련된 모든 자원들을 반환한다