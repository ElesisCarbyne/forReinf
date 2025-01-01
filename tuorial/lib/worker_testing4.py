def worker(num, bucket_tensor, event_count):
    event_count += 1
    while True:
        if event_count == 8:
            break
    print(f"Process {num}'s event_count : {event_count}")
    
    for i in range(10):
        bucket_tensor[num*10+i] = num*10+i