import math


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
        Dn = max(Dn, theoretical_func_res - empirical_function_result)
    Dn *= math.sqrt(n)
    return Dn < critical_value, Dn
