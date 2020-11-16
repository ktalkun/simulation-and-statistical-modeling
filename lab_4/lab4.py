import math

import matplotlib.pyplot as plt


def linear_congruential_generator(x, alpha, c, m):
    while True:
        x = (alpha * x + c) % m
        yield x / m


def normal_generator(mu, sigma, linear_gen):
    while True:
        u1 = next(linear_gen)
        u2 = next(linear_gen)
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2 * math.pi * u2)
        yield mu + z0 * sigma, mu + z1 * sigma


def integral_function(x):
    return (math.exp(-x ** 4) * math.sqrt(1 + x ** 4))


def double_integral_function(x, y):
    return (2 * math.pi) / ((1 + x) * (math.cos(2 * math.pi * y) ** 2 + (
            1 + x) ** 2 * math.sin(2 * math.pi * y) ** 4))


def cumulative_norm_distrib_func(value, mu, sigma):
    return 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(
        - (value - mu) ** 2 / (2 * sigma ** 2))


def calc_integral(x_sample, from_num, to_num):
    return sum(
        integral_function(x) / cumulative_norm_distrib_func(x, mu, sigma) for x
        in x_sample[from_num:to_num]) / (to_num - from_num)


def calc_double_integral(xy_sable, from_num, to_num):
    return sum(
        double_integral_function(x, y) for x, y in
        xy_sample_in_area[from_num: to_num]) / (to_num - from_num)


x0 = 79507
alpha0 = 79507
c = 63
m = 2 ** 31

mu = 0
sigma = 1

linear_gen = linear_congruential_generator(x0, alpha0, c, m)
normal_gen = normal_generator(mu, sigma, linear_gen)

# Task 1
exact_result = 2.000057
exampling_size = 1000000
x_sample = [next(normal_gen)[0] for _ in range(exampling_size)]

step = exampling_size // 100
steps = []
results = []

sum_res = 0
for size in range(step, exampling_size + 1, step):
    sum_res += calc_integral(x_sample, size - step, size)
    results.append(sum_res)
    steps.append(size)

results = [x / (i + 1) for x, i in zip(results, range(0, len(results)))]

discrepancy = [abs(x - exact_result) for x in results]
plt.plot(steps, discrepancy)
plt.show()
print("Task 1: " + str(results[len(results) - 1]))

# Task2
exact_result = 3.8579
exampling_size = 1000000
x_from = 0
x_to = 2 * math.pi
y_from = 1
y_to = 2

xy_sample = [next(normal_gen) for _ in range(exampling_size)]
xy_sample_in_area = list(
    filter(lambda xy: 0 < xy[0] < 1 and 0 < xy[1] < 1, xy_sample))

step = len(xy_sample_in_area) // 100  # 10 segments
steps = []
results = []

sum_res = 0
for size in range(step, len(xy_sample_in_area) + 1, step):
    sum_res += calc_double_integral(xy_sample_in_area, size - step, size)
    results.append(sum_res)
    steps.append(size)

results = [x / (i + 1) for x, i in zip(results, range(0, len(results)))]

discrepancy = [abs(x - exact_result) for x in results]
plt.plot(steps, discrepancy)
plt.show()
print("Task 2: " + str(results[len(results) - 1]))
