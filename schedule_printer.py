import numpy as np
import matplotlib.pylab as plt
from numpy import random

x_size = 30
y_size = 30


def read_schedule(sched_path):
    file = open(sched_path, 'r')
    sched = np.zeros((x_size, y_size), dtype=np.int32)
    for line in file.readlines():
        arr = line.split("\t")
        p = r2m(int(arr[0]))
        v = int(arr[1])
        sched[p[0]][p[1]] = v
    file.close()
    return sched


def show_schedule(schedules):
    for key in schedules.keys():
        sched = schedules[key]
        plt.figure()
        plt.imshow(sched, cmap="Pastel1")
        plt.title(key)
        plt.show()
        plt.close()
    pass


def r2m(idx):
    return idx // y_size, idx % y_size


if __name__ == '__main__':
    # Select schedules to view
    # schedule = read_schedule("resources\\default")
    schedule1 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule1.sched")
    schedule2 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule2.sched")
    schedule3 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule3.sched")
    schedules = dict()
    schedules["0"] = schedule1
    schedules["550"] = schedule2
    schedules["1050"] = schedule3
    show_schedule(schedules)