import torch
import torch.multiprocessing as mp

print("worker_testing2 is working now...")
def square(i, x):
    print("sub-sub-process is working...")
    x.pow_(2)
    print("In process {}: {}".format(i, x))

def worker(x, proc_num):
    print("sub-process is working...")
    processes = [] # 프로세스 풀 생성
    # mp.Pool()은 어떤 프로세스에게 어떤 작업을 배정할지를 자동으로 결정하는 반면,
    # mp.Process()는 어떤 프로세스에게 어떤 작업을 배정할지를 수동으로 지정한다
    # 프로세스 작업 수행 과정은 CMD에서 확인할 수 있다(.run() 메서드를 사용하면 직접 출력해 볼 수 있다)
    for i in range(proc_num):
        start_index = 8 * i
        # multiprocessing에서 process는 Process 객체가 생성되고 이 객체의 start() 메서드가 호출되었을 때 소환된다(spawn)
        # Process 객체는 분리된 프로세스 내에서 실행중인 활동을 나타낸다
        proc = mp.Process(target=square, args=(i, x[start_index:start_index+8])) # 프로세스 번호, 작업물, 결과물 저장 큐를 넘겨준다
        # proc.run() # 프로세스의 활동을 출력한다
                     # 이 메서드를 사용하지 않으면 (쥬피터의 경우) CMD에 프로세스 활동이 출력된다
        proc.start() # 프로세스를 시작한다(프로세스가 실제로 생성된다)
        processes.append(proc) # 추후 프로세스들을 제어하기 위해 프로세스들을 추적한다
    
    for proc in processes:
        proc.join() # join() 메서드를 호출한 프로세스가 종료될 때까지 주 프로세스가 해당 하위 프로세스를 기다린다(공식 문서는 이를 block이라고 표현하고 있다)
                    # 좀비 프로세스가 생성되는 것을 막기 위해 가급적이면 .join() 을 사용해서 프로세스를 종료시키는 것이 좋다
        proc.terminate() # 프로세스를 종료한다(자원 반환은 수행하지 않는다)
                         # 부모 프로세스를 종료하더라도 그 밑의 자식 프로세스들은 종료되지 않는다(이 경우 자식 프로세스들은 "기아"가 된다)
        proc.close() # 프로세스 객체를 해체하여 그것과 관련된 모든 자원들을 회수한다