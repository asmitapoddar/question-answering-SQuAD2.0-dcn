import time
import subprocess
import select
import math

import matplotlib.pyplot as plt
with open("loss.log", "r") as f:
    data = list(map(lambda s: tuple(s[:-1].split(": ")), f.readlines()))
    x = list(map(lambda d: int(d[0]), data))
    y = list(map(lambda d: math.log10(float(d[1])), data))
    print(x, y)
plt.plot(x, y, 'm-')
plt.pause(1)

filename = "loss.log"
f = subprocess.Popen(['tail','-F','-n 0',filename],\
        stdout=subprocess.PIPE,stderr=subprocess.PIPE)
p = select.poll()
p.register(f.stdout)

while True:
    if p.poll(1):
        line = f.stdout.readline()[:-1].decode("utf-8").split(": ")
        print(line)
        x += [int(line[0])]
        y += [math.log10(float(line[1]))]
        plt.plot(x, y, 'm-')
        plt.pause(0.1)
    time.sleep(1)
plt.show()
