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
