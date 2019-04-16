from deap import tools, base
from multiprocessing import Pool
from algorithms.ga_scheme import eaMuPlusLambda
from numpy import random as rnd
import numpy as np
from scipy.spatial import distance

from deap import creator
creator.create("TimeFit", base.Fitness, weights=(-1.0,))
creator.create("ScheduleMixtureIndividual", np.ndarray, fitness=creator.TimeFit)


class IntervalMixtureSchedGA:
    def individual(self):
        x_size = self.model.x_size
        y_size = self.model.y_size
        cores = self.model.cores
        max_centers = self.max_centers
        solution = rnd.random((cores, max_centers, 2)) * x_size # TODO if x==y
        return solution

    def mutation(self, mutant):
        core_idx = rnd.randint(self.model.cores)
        center_idx = rnd.randint(self.max_centers)
        cord_idx = rnd.randint(2)
        mutant[core_idx][center_idx][cord_idx] += rnd.normal(0, 0.3 * self.model.x_size) # TODO if x==y
        if mutant[core_idx][center_idx][cord_idx] < 0:
            mutant[core_idx][center_idx][cord_idx] = 0
        if mutant[core_idx][center_idx][cord_idx] > self.model.x_size:
            mutant[core_idx][center_idx][cord_idx] = self.model.x_size
        return mutant,

    def crossover(self, p1, p2):
        x_size = self.model.x_size
        y_size = self.model.y_size
        cores = self.model.cores
        max_centers = self.max_centers
        c1 = creator.ScheduleMixtureIndividual(np.zeros((cores, max_centers, 2)))
        c2 = creator.ScheduleMixtureIndividual(np.zeros((cores, max_centers, 2)))
        a, b = rnd.choice(range(cores), 2)
        a = np.minimum(a, b)
        b = np.maximum(a, b)
        i = 0
        while i < cores:
            if i < a or i > b:
                c1[i] = np.copy(p1[i])
                c2[i] = np.copy(p2[i])
            else:
                c1[i] = np.copy(p2[i])
                c2[i] = np.copy(p1[i])
            i+=1
        return c1, c2

    def calc_distance(self, point, centers):
        dists = np.zeros(len(centers))
        for c in range(len(centers)):
            dists[c] = distance.euclidean(point, centers[c])
        return dists.min()


    def solution_to_schedule(self, solution):
        x_size = self.model.x_size
        y_size = self.model.y_size
        cores = self.model.cores
        schedule = np.zeros((x_size, y_size), dtype=np.int32)
        for x in range(x_size):
            for y in range(y_size):
                distances = np.zeros(cores)
                for c in range(cores):
                    distances[c] = self.calc_distance((x, y), solution[c])
                core_idx = np.argmin(distances)
                schedule[x][y] = core_idx
        return schedule

    def evaluate_solution(self, solution):
        schedule = self.solution_to_schedule(solution)
        return self.model.simulation(schedule)


    def __init__(self, model, max_centers=1, ext_sol=None):
        # self.pool = Pool(5)
        # base params
        self.pop_size = 20
        self.generations = 100
        self.mut_prob = 0.6
        self.cross_prob = 0.1
        self.model = model
        self.max_centers = max_centers
        self.external_sol = ext_sol

        toolbox = base.Toolbox()
        # toolbox.register("map", self.pool.map)
        toolbox.register("map", map)

        toolbox.register("individual", tools.initIterate, creator.ScheduleMixtureIndividual, self.individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, self.pop_size)
        toolbox.register("mate", self.crossover)
        toolbox.register("mutate", self.mutation)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate_solution)

        self.toolbox = toolbox

    def __call__(self, start, end):
        self.model.ev_start = start
        self.model.ev_end = end
        pop = self.toolbox.population()
        if self.external_sol is not None:
            pop[0] = self.external_sol

        hof = tools.HallOfFame(3, np.array_equal)
        stats = tools.Statistics(lambda ind: np.array(ind.fitness.values))
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
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
    cores = 5
    # base_sched = creator.ScheduleIndividual(read_schedule("C:\\wspace\\projects\intmodel\\resources\\basic")) # remove external parameters
    ammodel = AgentMobilityModel(x_size, y_size, transportations, cores)
    scheduler = IntervalMixtureSchedGA(ammodel, max_centers=2, ext_sol=None)
    result = scheduler(1100, 1440)
    best_solution = result[2].items[0]
    best_schedule = scheduler.solution_to_schedule(best_solution)
    out_schedule = open("C:\\wspace\\projects\\intmodel\\tmp\\mix_schedule4.sched", 'w')
    for x in range(x_size):
        for y in range(y_size):
            out_schedule.write("{}\t{}\n".format(x * y_size + y, best_schedule[x][y]))
    out_schedule.close()


if __name__ == "__main__":
    main()