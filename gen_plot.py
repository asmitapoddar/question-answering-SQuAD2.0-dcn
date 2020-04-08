import time
import select
import subprocess
import sys
import math

import matplotlib.pyplot as plt

loss_path = "loss.log" if len(sys.argv) <= 1 else sys.argv[1]

HISTORY = 200
LOSS_SMOOTHING = 0  # Ballpark values: try setting to 5 for some smoothing or 20 for a lot of smoothing.

with open(loss_path, "r") as f:
    data = list(map(lambda s: tuple(s[:-1].split(": ")), f.readlines()[-HISTORY:]))
    data = sorted(list(filter(lambda tup: len(tup)==2, data)))
    x_loss = list(map(lambda d: int(d[0]), data))
    y_loss = list(map(lambda d: float(d[1]), data))

    if LOSS_SMOOTHING > 0:
        print("Using loss smoothing by running average of width %d." % (2*LOSS_SMOOTHING+1))
        y_loss_smoothed = []
        for i in range(len(y_loss)):
            ynew = 0.0
            count = 0
            for k in range(i-LOSS_SMOOTHING,i+LOSS_SMOOTHING+1):
                if 0 <= k < len(y_loss):
                    ynew += y_loss[k]
                    count += 1
            y_loss_smoothed.append(ynew/float(count))
        y_loss = y_loss_smoothed
        plt.title("(loss smoothing width %d)" % (2*LOSS_SMOOTHING+1))

plt.plot(x_loss, y_loss, 'm-')
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
        x_loss += [int(line[0])]
        y_loss += [float(line[1])]
        plt.plot(x_loss, y_loss, 'm-')
        plt.pause(0.1)
    time.sleep(1)
plt.show()
