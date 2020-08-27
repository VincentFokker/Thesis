from deap import base
from deap import creator
from deap import tools
from simple_conveyor_v9 import simple_conveyor

demand = [[2, 3, 1, 2, 3, 2, 2, 1, 3, 2], [3, 3, 2, 2, 2, 2, 3, 2, 2, 2], [1, 1, 3, 2, 2, 1, 1, 1, 2, 2], [1, 3, 1, 3, 1, 2, 1, 3, 3, 2], [1, 2, 1, 2, 3, 1, 3, 1, 1, 1]]
amount_gtp = 5
env = simple_conveyor(demand, amount_gtp, 3)


#Build action list
order_list = []
for index in range(len(demand[0])):
    order_list.append([item[index] for item in env.queues])
print(order_list)

def evaluate(individual):
    for item in individual:
        env.step(item)

        while env.demand_queues != [[] * i for i in range(amount_gtp)]:
        env.step(0)
    return env.total_travel

creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("evaluate", evaluate)


