import time
import datetime

tracking = time.time()

def values():
    global tracking

    now = datetime.datetime.now()
    cost = 100.00
    increase = .01

    newvalue = []

    for x in range(1,1000):
        print(x)
        time.sleep(2)

        if time.time() - 10 > tracking:
            newvalue.append(float(increase))
            print(newvalue)
            print(now)

            tracking = time.time()
            
values()