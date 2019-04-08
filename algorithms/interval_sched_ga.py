from deap import tools, base
from multiprocessing import Pool
from algorithms.ga_scheme import eaMuPlusLambda
from numpy import random as rnd
import numpy as np

from deap import creator
creator.create("TimeFit", base.Fitness, weights=(-1.0,))
creator.create("ScheduleIndividual", np.ndarray, fitness=creator.TimeFit)


class IntervalSchedGA:
    def individual(self):
        x_size = self.model.x_size
        y_size = self.model.y_size
        solution = rnd.randint(1, self.model.cores, size=(x_size, y_size))
        return solution

    def mutation(self, mutant):
        x_size = self.model.x_size
        y_size = self.model.y_size
        mutant[rnd.randint(x_size)][rnd.randint(y_size)] = rnd.randint(1, self.model.cores)
        return mutant,

    def crossover(self, p1, p2):
        x_size = self.model.x_size
        y_size = self.model.y_size
        c1 = creator.ScheduleIndividual(np.zeros((x_size, y_size), dtype=np.int32))
        c2 = creator.ScheduleIndividual(np.zeros((x_size, y_size), dtype=np.int32))
        for x in range(x_size):
            for y in range(y_size):
                if x < x_size / 2:
                    c1[x][y] = p1[x][y]
                    c2[x][y] = p2[x][y]
                else:
                    c1[x][y] = p2[x][y]
                    c2[x][y] = p1[x][y]
        return c1, c2

    def __init__(self, model, ext_sol=None):
        self.pool = Pool(6)
        # base params
        self.pop_size = 24
        self.generations = 1000
        self.mut_prob = 0.7
        self.cross_prob = 0.1
        self.model = model
        if ext_sol is not None:
            self.external_sol = ext_sol

        toolbox = base.Toolbox()
        toolbox.register("map", self.pool.map)

        toolbox.register("individual", tools.initIterate, creator.ScheduleIndividual, self.individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, self.pop_size)
        toolbox.register("mate", self.crossover)
        toolbox.register("mutate", self.mutation)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.model.simulation)

        self.toolbox = toolbox

    def __call__(self, start, end):
        self.model.ev_start = start
        self.model.ev_end = end
        pop = self.toolbox.population()
        pop[0] = self.external_sol

        hof = tools.HallOfFame(3, np.array_equal)
        stats = tools.Statistics(lambda ind: np.array(ind.fitness.values))
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        # stats.register("max", np.max)
        pop, logbook = eaMuPlusLambda(pop, self.toolbox, self.pop_size, self.pop_size, self.cross_prob, self.mut_prob,
                                      self.generations, stats=stats, halloffame=hof)
        return pop, logbook, hof


def main():
    from scenario_reader import read_transportations, read_schedule
    from model.model import AgentMobilityModel

    transportations_file = "C:\\wspace\\projects\\intmodel\\resources\\transportations"
    transportations = read_transportations(transportations_file)
    x_size = 30
    y_size = 30
    cores = 9
    base_sched = creator.ScheduleIndividual(read_schedule("C:\\wspace\\projects\intmodel\\resources\\basic")) # remove external parameters
    ammodel = AgentMobilityModel(x_size, y_size, transportations, cores)
    scheduler = IntervalSchedGA(ammodel, base_sched)
    result = scheduler(0, 1440)
    best = result[2].items[0]
    out_schedule = open("C:\\wspace\\projects\\intmodel\\tmp\\full_schedule", 'w')
    for x in range(x_size):
        for y in range(y_size):
            out_schedule.write("{}\t{}\n".format(x * y_size + y, best[x][y]))
    out_schedule.close()


if __name__ == "__main__":
    main()