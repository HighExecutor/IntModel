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
    # schedule = read_schedule("resources\\basic")
    # schedule1 = read_schedule("resources\\schedule1")
    # schedule2 = read_schedule("resources\\schedule2")
    # schedule3 = read_schedule("resources\\schedule3")
    schedule1 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule1.sched")
    schedule2 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule2.sched")
    schedule3 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule3.sched")
    # schedule1 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\last_schedule1")
    # schedule2 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\last_schedule2")
    # schedule3 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\last_schedule3")
    # full_ga_schedule = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\full_schedule")
    # full_ga_schedule = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\raw_spb_schedule")
    # full_spb_schedule = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule.sched")
    # schedule1 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule1_6.sched")
    # schedule2 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule2_6.sched")
    # schedule2 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule_velo_single.sched")
    # schedule3 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule3_6.sched")
    schedules = dict()
    schedules["0"] = schedule1
    schedules["550"] = schedule2
    schedules["1050"] = schedule3
    # schedules["basic"] = schedule
    # schedules['full'] = full_spb_schedule
    show_schedule(schedules)

    pass
