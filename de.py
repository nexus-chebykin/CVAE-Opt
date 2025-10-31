# ------------------------------------------------------------------------------+
# Based on the implementation from
# Nathan A. Rooy
# A simple, bare bones, implementation of differential evolution with Python
# August, 2017
#
# MIT License
#
# Copyright (c) 2017 Nathan Rooy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------+

from random import sample
import numpy as np
import time

def minimize(cost_func, args, search_space_bound, search_space_size, popsize, mutate, recombination, maxiter, maxtime, maxevaluations=None):

    # --- INITIALIZE A POPULATION (step #1) ----------------+
    start_time = time.time()
    population_cost = np.ones((popsize)) * np.inf
    children = np.zeros((popsize, search_space_size))
    iterations_without_improvement = 0
    gen_best = np.inf
    convergence_history = []
    time_history = []
    evaluations_done = 0  # Track total number of evaluations

    # Track timing breakdown
    ask_time_total = 0.0
    eval_time_total = 0.0
    tell_time_total = 0.0

    population = np.random.uniform(-search_space_bound, search_space_bound,
                                   (popsize, search_space_size))

    # --- SOLVE --------------------------------------------+

    # cycle through each generation (step #2)
    iteration = 0
    while True:
        iteration += 1

        # Check stopping criteria
        # Always check time limit
        if time.time() - start_time > maxtime:
            break
        # Only check iteration limit if specified (overrides time)
        if maxiter is not None and iteration > maxiter:
            break
        # Only check evaluation limit if specified (overrides time)
        if maxevaluations is not None and evaluations_done >= maxevaluations:
            break

        # --- ASK: Generate candidate solutions ----------------+
        ask_start = time.time()

        # cycle through each individual in the population
        for j in range(0, popsize):
            # --- MUTATION (step #3.A) ---------------------+

            # select three random vector index positions [0, popsize), not including current vector (j)
            candidates = list(range(0, popsize))
            candidates.remove(j)
            random_index = sample(candidates, 3)

            # subtract x3 from x2, and create a new vector (x_diff)
            x_diff = population[random_index[1]] - population[random_index[2]]

            # multiply x_diff by the mutation factor (F) and add to x_1
            child = population[random_index[0]] + mutate * x_diff

            # --- RECOMBINATION (step #3.B) ----------------+
            crossover = np.random.uniform(0, 1, search_space_size)
            crossover = crossover > recombination
            child[crossover] = population[j][crossover]

            children[j] = child

        # Ensure bounds
        children = np.clip(children, -search_space_bound, search_space_bound)

        ask_time_total += time.time() - ask_start

        # --- EVALUATE: Objective function evaluation ------+
        eval_start = time.time()
        _, scores_trial = cost_func(children, *args)
        scores_trial = np.array(scores_trial)
        evaluations_done += popsize  # Increment evaluation counter
        eval_time_total += time.time() - eval_start

        # --- TELL: Update population with results ---------+
        tell_start = time.time()

        iterations_without_improvement += 1
        if min(population_cost) > min(scores_trial):
            iterations_without_improvement = 0

        improvement = population_cost > scores_trial
        population[improvement] = children[improvement]
        population_cost[improvement] = scores_trial[improvement]

        tell_time_total += time.time() - tell_start

        # --- SCORE KEEPING --------------------------------+
        gen_best = min(population_cost)  # fitness of best individual
        convergence_history.append(gen_best)
        time_history.append(time.time() - start_time)

    # Return timing breakdown
    timing_breakdown = {
        'ask_time': ask_time_total,
        'eval_time': eval_time_total,
        'tell_time': tell_time_total
    }

    return gen_best, population[np.argmin(population_cost)], convergence_history, time_history, timing_breakdown

