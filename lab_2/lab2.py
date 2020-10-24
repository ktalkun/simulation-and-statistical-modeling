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


def geometric_generator(p, linear_gen):
    while True:
        yield math.floor(math.log(next(linear_gen)) / math.log(1 - p))


def bernoulli_generator(p, linear_gen):
    while True:
        yield int(next(linear_gen) <= p)


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


def geometric_distribution_func(p, unique_x_geometric, value):
    return (1 - p) ** unique_x_geometric.index(value) * p


def bernoulli_distribution_func(p, value):
    return p if value == 1 else 1 - p


x0 = 79507
alpha0 = 79507
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

# POISSON SAMPLE, LAMBDA = 1
l = 1
poisson_gen = poisson_generator(l, linear_congruential_generator(x0, alpha0,
                                                                 0, m))
x_poisson = [next(poisson_gen) for _ in range(1000)]
# print('\n'.join(map(str, x_poisson)))

unique_x_poisson = sorted(list(Counter(x_poisson).keys()))
# Кол-во степеней свободы (для 10 варианта 6 - 1 = 5 степеней свободы)
k_poisson = len(unique_x_poisson)
critical_value_poisson = 11.07
hi_squa_test2 = hi_squared_test(x_poisson,
                                partial(poisson_distribution_func, l),
                                critical_value_poisson)
print('Poisson generator, lambda = 1:')
print('Hi Squared Pirson criteria: ' + str(hi_squa_test2[1]) + ' <= '
      + str(critical_value_poisson) if hi_squa_test2[0] else
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
plt.title('Poisson generator, $\lambda = 1$')
plt.show()

# GEOMETRIC SAMPLE.
p = 0.2
geometric_gen = geometric_generator(p, linear_congruential_generator(x0, alpha0,
                                                                     0, m))
x_geometric = [next(geometric_gen) for _ in range(1000)]
# print('\n'.join(map(str, x_geometric)))

unique_x_geometric = sorted(list(Counter(x_geometric).keys()))
# Кол-во степеней свободы (для 10 варианта 27 - 1 = 26 степеней свободы)
k_geometric = len(unique_x_geometric)
critical_value_geometric = 38.89
hi_squa_test3 = hi_squared_test(x_geometric,
                                partial(geometric_distribution_func, p,
                                        unique_x_geometric),
                                critical_value_geometric)
print('Geometric generator, p = 1:')
print('Hi Squared Pirson criteria: ' + str(hi_squa_test3[1]) + ' <= '
      + str(critical_value_geometric) if hi_squa_test3[0] else
      'Zero hypothesis fails by Hi Squared Pirson criteria.')
theoretical_expectation = 1 / p
empirical_dispersion = empirical_dispersion_func(x_geometric)
theoretical_dispersion = (1 - p) / p ** 2
empirical_expectation = empirical_expectation_func(x_geometric)
print('theoretical expectation: ', theoretical_expectation)
print('empirical expectation: ', empirical_expectation)
print('theoretical dispersion: ', theoretical_dispersion)
print('empirical dispersion: ', empirical_dispersion)
print('')

unique_x_geometric.append(unique_x_geometric[k_geometric - 1] + 1)
plt.hist(x_geometric, bins=sorted(list(unique_x_geometric)), ec='#666633',
         facecolor="#99ff33")
plt.title('Geometric generator,  $p = 0.2$')
plt.show()

# BERNOULLI SAMPLE.
p = 0.75
bernoulli_gen = bernoulli_generator(p, linear_congruential_generator(x0, alpha0,
                                                                     0, m))
x_bernoulli = [next(bernoulli_gen) for _ in range(1000)]
# print('\n'.join(map(str, x_bernoulli)))

critical_x_bernoulli = 10
unique_x_bernoulli = sorted(list(Counter(x_bernoulli).keys()))
# Кол-во степеней свободы (для 10 варианта 2 - 1 = 1 степеней свободы)
k_bernoulli = len(unique_x_bernoulli)
critical_value_bernoulli = 3.841
hi_squa_test4 = hi_squared_test(x_bernoulli,
                                partial(bernoulli_distribution_func, p),
                                critical_value_bernoulli)
print('Geometric generator, p = 1:')
print('Hi Squared Pirson criteria: ' + str(hi_squa_test4[1]) + ' <= '
      + str(critical_value_bernoulli) if hi_squa_test4[0] else
      'Zero hypothesis fails by Hi Squared Pirson criteria.')

theoretical_expectation = p
empirical_dispersion = empirical_dispersion_func(x_poisson)
theoretical_dispersion = p * (1 - p)
empirical_expectation = empirical_expectation_func(x_poisson)
print('theoretical expectation: ', theoretical_expectation)
print('empirical expectation: ', empirical_expectation)
print('theoretical dispersion: ', theoretical_dispersion)
print('empirical dispersion: ', empirical_dispersion)

unique_x_bernoulli.append(unique_x_bernoulli[k_bernoulli - 1] + 1)
plt.hist(x_bernoulli, bins=unique_x_bernoulli, ec='#666633',
         facecolor="#99ff33")
plt.title('Bernoulli generator,  $p = 0.75$')
plt.show()
