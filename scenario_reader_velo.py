import numpy as np
from model.model import AgentMobilityModel

x_size = 30
y_size = 30


def read_iteration(file):
    corr_count = int(file.readline())
    corr = np.zeros((corr_count, 4), dtype=np.float32)
    for i in range(corr_count):
        arr = file.readline().split('\t')
        corr[i][0] = int(arr[0])
        corr[i][1] = int(arr[1])
        corr[i][2] = int(arr[2])
        corr[i][3] = float(arr[3])
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


def corresponds_modelling_velo(transportations, schedules, cores):
    ammodel = AgentMobilityModel(x_size, y_size, transportations, cores)
    ammodel.interactive_simulation2_velo(schedules)

def main():
    transportations_file = "resources\\spb_passengers_center_100k_5"
    transportations = read_transportations(transportations_file)
    cores = 9
    # schedule = read_schedule("resources\\basic")
    # schedule1 = read_schedule("resources\\schedule1")
    # schedule2 = read_schedule("resources\\schedule2")
    # schedule3 = read_schedule("resources\\schedule3")
    # full_spb_schedule = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule.sched")
    # schedule1 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule1.sched")
    # schedule2 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule2.sched")
    # schedule3 = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule3.sched")
    # schedule_single = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\spb_schedule_velo_single.sched")
    schedules = dict()
    # schedules[0] = schedule
    # schedules[0] = full_spb_schedule
    # schedules[0] = schedule_single
    # schedules[0] = schedule1
    # schedules[530] = schedule2
    # schedules[1050] = schedule3
    for iters in range(10):
        iter_sched = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\multiple\\10_include\\spb_schedule_velo_{}_{}.sched".format(iters*144, (iters+1)*144))
        schedules[iters*144] = iter_sched
    # for iters in range(5):
    #     iter_sched = read_schedule("C:\\wspace\\projects\\intmodel\\tmp\\multiple\\5_include\\spb_schedule_velo_{}_{}.sched".format(iters*288, (iters+1)*288))
    #     schedules[iters*288] = iter_sched
    result = corresponds_modelling_velo(transportations, schedules, cores)


    pass
