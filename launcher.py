from algorithms.interval_sched_ga import IntervalSchedGA

if __name__ == "__main__":
    from scenario_reader import read_transportations
    from model.model import AgentMobilityModel
    import algorithms.interval_sched_ga

    algorithms.interval_sched_ga.main()
