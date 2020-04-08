import time
import select
import subprocess
import sys
import math

import matplotlib.pyplot as plt

loss_path = "loss.log" if len(sys.argv) <= 1 else sys.argv[1]

HISTORY = 15000
SMOOTHING = 0  # Ballpark values: try setting to 5 for some smoothing or 20 for a lot of smoothing.

with open(loss_path, "r") as f:
    data = list(map(lambda s: tuple(s[:-1].split(": ")), f.readlines()[-HISTORY:]))
    data = sorted(list(filter(lambda tup: len(tup)==2, data)))
    x = list(map(lambda d: int(d[0]), data))
    y = list(map(lambda d: float(d[1]), data))

    if SMOOTHING > 0:
        print("Using smoothing by running average of width %d." % (2*SMOOTHING+1))
        for i in range(len(y)):
            ynew = 0.0
            count = 0
            for k in range(i-SMOOTHING,i+SMOOTHING+1):
                if 0 <= k < len(y):
                    ynew += y[k]
                    count += 1
            y[i] = ynew/float(count)

    #print(x, y)
plt.plot(x, y, 'm-')
plt.xlabel("Number of Steps")
plt.ylabel("Loss")
plt.title("Training curve of " + loss_path)
plt.pause(1)

filename = loss_path
f = subprocess.Popen(['tail','-F','-n 0',filename],\
        stdout=subprocess.PIPE,stderr=subprocess.PIPE)
p = select.poll()
p.register(f.stdout)

while True:
    if p.poll(1):
        line = f.stdout.readline()[:-1].decode("utf-8").split(": ")
        print(line)
        x += [int(line[0])]
        y += [float(line[1])]
        plt.plot(x, y, 'm-')
        plt.pause(0.1)
    time.sleep(1)
plt.show()
