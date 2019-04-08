import numpy as np
from numpy import random as rnd
from deap import creator

agent_model_time = 0.1
agent_transfer_time = 0.5


def agent_model_time_func(agents):
    stable_time = agents * agent_model_time
    return stable_time + rnd.normal(loc=stable_time * 0.1, scale=stable_time * 0.01)


v_agent_transfer_time = np.vectorize(agent_model_time_func)


class AgentMobilityModel:
    def __init__(self, x_size, y_size, transportations, cores):
        self.x_size = x_size
        self.y_size = y_size
        self.transportations = transportations
        self.iterations = len(transportations)
        self.cores = cores
        self.ev_start = 0
        self.ev_end = self.iterations

    def r2m(self, idx):
        return idx // self.y_size, idx % self.y_size

    def simulation(self, schedule):
        # initialization
        # model_times = np.zeros(self.iterations)
        # transfer_times = np.zeros(self.iterations)
        total_times = np.zeros(self.ev_end)
        schedule = schedule
        field = np.zeros((self.x_size, self.y_size), dtype=np.int32)

        for i in range(self.ev_end):
            # iteration
            iteration_data = self.transportations[i]
            cores_model = np.zeros(self.cores)
            cores_transfers = np.zeros(self.cores)

            for corr in iteration_data:
                if corr[0] == corr[1]:
                    p = self.r2m(corr[0])
                    field[p[0]][p[1]] += corr[2]
                else:
                    p1 = self.r2m(corr[0])
                    p2 = self.r2m(corr[1])
                    field[p1[0]][p1[1]] -= corr[2]
                    field[p2[0]][p2[1]] += corr[2]
                    core1 = schedule[p1[0]][p1[1]]
                    core2 = schedule[p2[0]][p2[1]]
                    if core1 != core2:
                        cores_transfers[core1 - 1] += corr[2]
                        cores_transfers[core2 - 1] += corr[2]
            for x in range(self.x_size):
                for y in range(self.y_size):
                    cores_model[schedule[x][y] - 1] += field[x][y]

            cores_transfers = cores_transfers * agent_transfer_time
            cores_model = v_agent_transfer_time(cores_model)
            total_model_time = cores_model + cores_transfers

            # model_times[i] = cores_model.max()
            # transfer_times[i] = cores_transfers.max()
            total_times[i] = total_model_time.max()
        result = total_times[self.ev_start:self.ev_end].sum()
        return result,
