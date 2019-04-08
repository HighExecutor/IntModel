import numpy as np
import matplotlib.pylab as plt
from sys import stdout
from numpy import random

x_size = 30
y_size = 30
agent_model_time = 0.1
agent_transfer_time = 0.01


def agent_model_time_func(agents):
    stable_time = agents * agent_model_time
    return stable_time + random.normal(loc=stable_time*0.1, scale=stable_time*0.01)

v_agent_transfer_time = np.vectorize(agent_model_time_func)

def agent_model_test():
    ag = np.arange(10000, 1000000, 100)
    times = v_agent_transfer_time(ag)
    plt.plot(ag, times)
    plt.plot(ag, ag * agent_model_time)
    plt.show()

def read_iteration(file):
    corr_count = int(file.readline())
    corr = np.zeros((corr_count, 3), dtype=np.int32)
    for i in range(corr_count):
        arr = file.readline().split('\t')
        corr[i][0] = int(arr[0])
        corr[i][1] = int(arr[1])
        corr[i][2] = int(arr[2])
    return corr


def read_transportations(path):
    file = open(path, 'r')
    iters = int(file.readline())
    transportations = list()
    iter = 0
    while iter < iters:
        transportations.append(read_iteration(file))
        iter += 1
    file.close()
    return transportations


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


def r2m(idx):
    return idx // y_size, idx % y_size


def corresponds_modelling(transportations, schedules):
    # initialization
    iters = len(transportations)
    model_times = np.zeros(iters)
    transfer_times = np.zeros(iters)
    total_times = np.zeros(iters)
    schedule = schedules[0]
    field = np.zeros((x_size, y_size), dtype=np.int32)
    plt.ion()
    plt.imshow(field)
    plt.contour(schedule, alpha=0.5, cmap='Set1')
    plt.tight_layout()
    plt.show()
    plt.pause(0.0000001)

    for i in range(iters):
        if i in schedules.keys():
            schedule = schedules[i]
        # iteration
        cores = schedule.max()
        iter = transportations[i]
        cores_model = np.zeros(cores)
        cores_transfers = np.zeros(cores)

        for corr in iter:
            if corr[0] == corr[1]:
                p = r2m(corr[0])
                field[p[0]][p[1]] += corr[2]
            else:
                p1 = r2m(corr[0])
                p2 = r2m(corr[1])
                field[p1[0]][p1[1]] -= corr[2]
                field[p2[0]][p2[1]] += corr[2]
                core1 = schedule[p1[0]][p1[1]]
                core2 = schedule[p2[0]][p2[1]]
                if core1 != core2:
                    cores_transfers[core1-1] += corr[2]
                    cores_transfers[core2-1] += corr[2]
        for x in range(x_size):
            for y in range(y_size):
                cores_model[schedule[x][y]-1] += field[x][y]

        cores_transfers = cores_transfers * agent_transfer_time
        cores_model = v_agent_transfer_time(cores_model)
        total_model_time = cores_model + cores_transfers

        model_times[i] = cores_model.max()
        transfer_times[i] = cores_transfers.max()
        total_times[i] = total_model_time.max()

        # draw output
        if i % 25 == 0:
            plt.imshow(field)
            plt.contour(schedule, cmap='Set1', alpha=0.4, linestyles="--")
            plt.show()
            plt.pause(0.0000001)
            plt.clf()
        stdout.write("\riteration=%d" % i)
        stdout.flush()
    plt.close()
    plt.ioff()
    print("\nmodel sum={}".format(model_times.sum()))
    print("transfer sum={}".format(transfer_times.sum()))
    print("total time={}".format(total_times.sum()))
    x_range = np.arange(0, iters)
    plt.plot(x_range, model_times, label='model')
    plt.plot(x_range, transfer_times, label='transfer')
    plt.plot(x_range, total_times, label='total')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # agent_model_test()
    transportations_file = "resources\\transportations"
    transportations = read_transportations(transportations_file)
    # schedule = read_schedule("resources\\basic")
    schedule1 = read_schedule("resources\\schedule1")
    schedule2 = read_schedule("resources\\schedule2")
    schedule3 = read_schedule("resources\\schedule3")
    # schedule1 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\last_schedule1")
    # schedule2 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\last_schedule2")
    # schedule3 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\last_schedule3")
    # full_ga_schedule = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\full_schedule")
    schedules = dict()
    # schedules[0] = schedule
    # schedules[0] = full_ga_schedule
    schedules[0] = schedule1
    schedules[500] = schedule2
    schedules[1100] = schedule3
    corresponds_modelling(transportations, schedules)

    pass
