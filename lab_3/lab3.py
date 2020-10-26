import math
import random
from functools import partial

import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, kstwobign, lognorm


def linear_congruential_generator(x, alpha, c, m):
    while True:
        x = (alpha * x + c) % m
        yield x / m


def normal_generator(mu, sigma, linear_gen):
    while True:
        u1 = next(linear_gen)
        u2 = next(linear_gen)
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
        # z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2 * math.pi * u2)
        yield mu + z0 * sigma


def lognormal_generator(mu, sigma, linear_gen):
    normal_gen = normal_generator(mu, sigma, linear_gen)
    while True:
        yield math.exp(next(normal_gen))


def hi_squared_test(frequencies, borders, distribution_func, p_value):
    exampling_size = sum(frequencies)
    hi_squared = 0
    for i in range(1, len(frequencies) + 1):
        empiric_freq = frequencies[i - 1]
        theoretic_freq = (distribution_func(borders[i]) - distribution_func(
            borders[i - 1])) * exampling_size
        if theoretic_freq:
            hi_squared += ((
                                   empiric_freq - theoretic_freq) ** 2) / theoretic_freq
    degrees_of_freedom = len(frequencies) - 1
    critical_value = chi2.ppf(1 - p_value, degrees_of_freedom)
    return hi_squared < critical_value, hi_squared, critical_value


def kolmogorov_test(values, distribution_func, p_value):
    values.sort()
    Dn = 0
    i = 0
    n = len(values)
    for value in values:
        i += 1
        theoretical_func_res = distribution_func(value)
        empirical_function_res = i / n
        Dn = max(Dn, abs(theoretical_func_res - empirical_function_res))
    Dn *= math.sqrt(n)
    critical_value = kstwobign.ppf(1 - p_value)
    return Dn < critical_value, Dn, critical_value


def empirical_expectation_func(values):
    return sum(values) / len(values)


def empirical_dispersion_func(values):
    expectation = empirical_expectation_func(values)
    result = 0
    for value in values:
        result += (value - expectation) ** 2
    return result / len(values) - 1


def cumulative_norm_distrib_func(mu, sigma, value):
    return norm.cdf(value, mu, sigma)


def cumulative_lognorm_distrib_func(mu, sigma, value):
    return lognorm.cdf(value, scale=math.exp(mu), s=sigma)


def built_in_random():
    while True:
        yield random.random()


x0 = 79507
alpha0 = 79507
c = 63
m = 2 ** 31
p_value = 0.05

generator = linear_congruential_generator(x0, alpha0, c, m)

# NORMAL SAMPLE, MU = 1, SIGMA^2 = 9
mu = 1
sigma = 3

normal_gen = normal_generator(mu, sigma, generator)
x_normal = [next(normal_gen) for _ in range(1000)]
# print('\n'.join(map(str, x_normal)))

freq_normal, borders_normal, _ = plt.hist(x_normal, bins='auto',
                                          ec='#666633',
                                          facecolor="#99ff33")
plt.title('Normal generator,  $\mu = 1, \sigma^2=9$')
plt.show()

hi_squa_test1 = hi_squared_test(freq_normal,
                                borders_normal,
                                partial(cumulative_norm_distrib_func,
                                        mu, sigma),
                                p_value)
print('Normal generator, mu = 1, sigma^2 = 9:')
print('Hi Squared Pirson criteria: ' + str(hi_squa_test1[1]) + ' <= '
      + str(hi_squa_test1[2]) if hi_squa_test1[0] else
      'Zero hypothesis fails by Hi Squared Pirson criteria.')
kolm_test1 = kolmogorov_test(x_normal,
                             partial(cumulative_norm_distrib_func, mu, sigma),
                             p_value)
print('Kolmogorov criteria: ' + str(kolm_test1[1]) + ' <= '
      + str(kolm_test1[2]) if kolm_test1[0]
      else 'Zero hypothesis fails by Kolmogorov criteria.')

theoretical_expectation = mu
empirical_dispersion = empirical_dispersion_func(x_normal)
theoretical_dispersion = sigma ** 2
empirical_expectation = empirical_expectation_func(x_normal)
print('theoretical expectation: ', theoretical_expectation)
print('empirical expectation: ', empirical_expectation)
print('theoretical dispersion: ', theoretical_dispersion)
print('empirical dispersion: ', empirical_dispersion)
print('')

# NORMAL SAMPLE, MU = 0, SIGMA^2 = 1
mu = 0
sigma = 1

normal_gen = normal_generator(mu, sigma, generator)
x_normal = [next(normal_gen) for _ in range(1000)]
# print('\n'.join(map(str, x_normal)))

freq_normal, borders_normal, _ = plt.hist(x_normal, bins='auto',
                                          ec='#666633',
                                          facecolor="#99ff33")
plt.title('Normal generator,  $\mu = 0, \sigma^2=1$')
plt.show()

hi_squa_test2 = hi_squared_test(freq_normal,
                                borders_normal,
                                partial(cumulative_norm_distrib_func,
                                        mu, sigma),
                                p_value)
print('Normal generator, mu = 0, sigma^2 = 1:')
print('Hi Squared Pirson criteria: ' + str(hi_squa_test2[1]) + ' <= '
      + str(hi_squa_test2[2]) if hi_squa_test2[0] else
      'Zero hypothesis fails by Hi Squared Pirson criteria.')
kolm_test2 = kolmogorov_test(x_normal,
                             partial(cumulative_norm_distrib_func, mu, sigma),
                             p_value)
print('Kolmogorov criteria: ' + str(kolm_test2[1]) + ' <= '
      + str(kolm_test2[2]) if kolm_test2[0]
      else 'Zero hypothesis fails by Kolmogorov criteria.')

theoretical_expectation = mu
empirical_dispersion = empirical_dispersion_func(x_normal)
theoretical_dispersion = sigma ** 2
empirical_expectation = empirical_expectation_func(x_normal)
print('theoretical expectation: ', theoretical_expectation)
print('empirical expectation: ', empirical_expectation)
print('theoretical dispersion: ', theoretical_dispersion)
print('empirical dispersion: ', empirical_dispersion)
print('')

# LOGNORMAL SAMPLE, MU = 1, SIGMA^2 = 9
mu = 1
sigma = 3

lognormal_gen = lognormal_generator(mu, sigma, generator)
x_lognormal = [next(lognormal_gen) for _ in range(1000)]
# print('\n'.join(map(str, x_lognormal)))

freq_lognormal, borders_lognormal, _ = plt.hist(x_lognormal, bins='auto',
                                                ec='#666633',
                                                facecolor="#99ff33")
plt.title('Lognormal generator,  $\mu = 1, \sigma^2=9$')
plt.show()

hi_squa_test3 = hi_squared_test(freq_lognormal,
                                borders_lognormal,
                                partial(cumulative_norm_distrib_func,
                                        mu, sigma),
                                p_value)

print('Lognormal generator, mu = 1, sigma^2 = 9:')
print('Hi Squared Pirson criteria: ' + str(hi_squa_test3[1]) + ' <= '
      + str(hi_squa_test3[2]) if hi_squa_test3[0] else
      'Zero hypothesis fails by Hi Squared Pirson criteria.')
kolm_test3 = kolmogorov_test(x_lognormal,
                             partial(cumulative_lognorm_distrib_func, mu,
                                     sigma),
                             p_value)
print('Kolmogorov criteria: ' + str(kolm_test3[1]) + ' <= '
      + str(kolm_test3[2]) if kolm_test3[0]
      else 'Zero hypothesis fails by Kolmogorov criteria.')

theoretical_expectation = math.exp(mu + sigma ** 2 / 2)
empirical_dispersion = empirical_dispersion_func(x_lognormal)
theoretical_dispersion = (math.exp(sigma ** 2) - 1) * math.exp(
    2 * mu + sigma ** 2)
empirical_expectation = empirical_expectation_func(x_lognormal)
print('theoretical expectation: ', theoretical_expectation)
print('empirical expectation: ', empirical_expectation)
print('theoretical dispersion: ', theoretical_dispersion)
print('empirical dispersion: ', empirical_dispersion)
print('')
