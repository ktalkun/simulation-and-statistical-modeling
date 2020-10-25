import math

import matplotlib.pyplot as plt


def linear_congruential_generator(x, alpha, c, m):
    while (True):
        x = (alpha * x + c) % m
        yield x / m


def multiplexial_congruential_generator(x, alpha, m):
    generator = linear_congruential_generator(x, alpha, 0, m)
    while (True):
        yield next(generator)


def mclaren_marsaglia_generator(x_generator, y_generator, k):
    V = [next(x_generator) for _ in range(k)]
    while (True):
        X = next(x_generator)
        Y = next(y_generator)
        j = math.floor(k * Y)
        yield V[j]
        V[j] = X


def hi_squared_test(values, k, critical_value):
    nu = [0] * k
    for value in values:
        nu[math.floor(value * k)] += 1
    p_k = len(values) / k
    hi_squared = 0
    for value in nu:
        hi_squared += ((value - p_k) ** 2) / p_k
    #     Для уровня значимости 0.05 при 9-ти степенях свободы.
    return hi_squared < critical_value, hi_squared


def kolmogorov_test(values, critical_value):
    values.sort()
    Dn = 0
    i = 0
    n = len(values)
    for value in values:
        i += 1
        # F(X) = (x-a)/(b-a) = [для a = 0 и b = 1] = x.
        theoretical_func_res = value
        # кол-во значение в выборке меньших текущего значения из выборки.
        empirical_function_result = i / n
        Dn = max(Dn, abs(theoretical_func_res - empirical_function_result))
    Dn *= math.sqrt(n)
    return Dn < critical_value, Dn


x0 = 16807
alpha0 = 16807
K = 64

x1 = 8195
alpha1 = 8195
c = 46
k = 64

m = 2 ** 31

hi_squared_critical_value = 16.919
kolmogorov_critical_value = 1.359

mult_congr_gen = multiplexial_congruential_generator(x0, alpha0, m)
x = [next(mult_congr_gen) for _ in range(1000)]
# print('\n'.join(map(str, x)))
hi_squa_test1 = hi_squared_test(x, 10, hi_squared_critical_value)
kolm_test1 = kolmogorov_test(x, kolmogorov_critical_value)

print('Multiplexial congruential generator:')
print('Hi Squared Pirson criteria: ' + str(hi_squa_test1[1]) + ' <= '
      + str(hi_squared_critical_value) if hi_squa_test1[0] else
      'Zero hypothesis fails by Hi Squared Pirson criteria.')
print('Kolmogorov criteria: ' + str(kolm_test1[1]) + ' <= '
      + str(kolmogorov_critical_value) if kolm_test1[0]
      else 'Zero hypothesis fails by Kolmogorov criteria.')

plt.hist(x, 10, ec='#993300', facecolor='#ff9900')
plt.title('Multiplexial congruential generator')
plt.show()

x = linear_congruential_generator(x0, alpha0, 0, m)
y = linear_congruential_generator(x1, alpha1, c, m)
mclar_mars_gen = mclaren_marsaglia_generator(x, y, k)
z = [next(mclar_mars_gen) for _ in range(1000)]
# print('\n'.join(map(str, z)))
hi_squa_test2 = hi_squared_test(z, 10, hi_squared_critical_value)
kolm_test2 = kolmogorov_test(z, kolmogorov_critical_value)

print('\nMcLaren marsaglia generator:')
print('Hi Squared Pirson criteria: ' + str(hi_squa_test2[1]) + ' <= '
      + str(hi_squared_critical_value) if hi_squa_test2[0]
      else 'Zero hypothesis fails by Hi Squared Pirson criteria.')
print('Kolmogorov criteria: ' + str(kolm_test2[1]) + ' <= '
      + str(kolmogorov_critical_value) if kolm_test2[0]
      else 'Zero hypothesis fails by Kolmogorov criteria.')

plt.hist(z, 10, ec='#666633', facecolor="#99ff33")
plt.title('McLaren marsaglia generator')
plt.show()
