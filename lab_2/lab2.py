import math
from collections import Counter
from functools import partial

import matplotlib.pyplot as plt


def linear_congruential_generator(x, alpha, c, m):
    while True:
        x = (alpha * x + c) % m
        yield x / m


def poisson_generator(l, linear_gen):
    d = math.exp(-l)
    while True:
        x = 1
        j = 0
        while x > d:
            x *= next(linear_gen)
            j += 1
        yield j - 1


def hi_squared_test(values, distribution_func, critical_value):
    distinct_map = Counter(values).most_common()
    exampling_size = len(values)
    hi_squared = 0
    for pair in distinct_map:
        empiric_freq = pair[1]
        random_value = pair[0]
        theoretic_freq = math.ceil(
            exampling_size * distribution_func(random_value))
        hi_squared += ((empiric_freq - theoretic_freq) ** 2) / theoretic_freq
    #     Для уровня значимости 0.05 при 9-ти степенях свободы.
    return hi_squared < critical_value, hi_squared


def empirical_expectation_func(values):
    return sum(values) / len(values)


def empirical_dispersion_func(values):
    expectation = empirical_expectation_func(values)
    result = 0
    for value in values:
        result += (value - expectation) ** 2
    return result / len(values) - 1


def poisson_distribution_func(l, value):
    return l ** value * math.exp(-l) / math.factorial(value)


x0 = 79507
alpha0 = 79507
K = 64
m = 2 ** 31

# POISSON SAMPLE, LAMBDA = 0.7.
l = 0.7
poisson_gen = poisson_generator(l, linear_congruential_generator(x0, alpha0,
                                                                 0, m))
x_poisson = [next(poisson_gen) for _ in range(1000)]
# print('\n'.join(map(str, x_poisson)))

unique_x_poisson = sorted(list(Counter(x_poisson).keys()))
# Кол-во степеней свободы (для 10 варианта 6 - 1 = 5 степеней свободы)
k_poisson = len(unique_x_poisson)
critical_value_poisson = 11.07
hi_squa_test1 = hi_squared_test(x_poisson,
                                partial(poisson_distribution_func, l),
                                critical_value_poisson)
print('Poisson generator, lambda = 0.7:')
print('Hi Squared Pirson criteria: ' + str(hi_squa_test1[1]) + ' <= '
      + str(critical_value_poisson) if hi_squa_test1[0] else
      'Zero hypothesis fails by Hi Squared Pirson criteria.')

theoretical_expectation = l
empirical_dispersion = empirical_dispersion_func(x_poisson)
theoretical_dispersion = l
empirical_expectation = empirical_expectation_func(x_poisson)
print('theoretical expectation: ', theoretical_expectation)
print('empirical expectation: ', empirical_expectation)
print('theoretical dispersion: ', theoretical_dispersion)
print('empirical dispersion: ', empirical_dispersion)
print('')

unique_x_poisson.append(unique_x_poisson[k_poisson - 1] + 1)
plt.hist(x_poisson, bins=unique_x_poisson, ec='#666633',
         facecolor="#99ff33")
plt.title('Poisson generator,  $\lambda = 0.7$')
plt.show()
