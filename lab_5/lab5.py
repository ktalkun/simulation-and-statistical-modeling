import math
import random

import matplotlib.pyplot as plt
import numpy as np

# Длина цепи Маркова
min_chain_length = 100
max_chain_length = 1000
# Шаг
length_step = 100
# Количество реализаций цепи Маркова
min_chain_count = 1000
max_chain_count = 10000
# Шаг
count_step = 1000

# Исходная матрица
g_A_real = ((1.2, 0.1, -0.3), (-0.3, 0.9, -0.2), (0.4, 0.5, 1),)
# Преобразованная матрица
g_A = ((-0.2, -0.1, 0.3), (0.3, 0.1, 0.2), (-0.4, -0.5, 0),)
# Правая часть системы
g_f = (2, 3, 3)


def built_in_random():
    while True:
        yield random.random()


generator = built_in_random()


def solve(A, f, chain_length, chain_count):
    # Размерность системы
    n = len(f)
    # Решение системы
    X = [0.0] * n

    # Вектор нач. вероятностей цепи Маркова
    pi = [1 / n] * n
    # Матрица переходных состояний цепи Маркова
    P = [[1 / n] * n] * n

    # Веса состояний цепи Маркова
    Q = [0.0] * (chain_length + 1)

    # СВ
    ksi = [0.0] * chain_count
    # БСВ
    alpha = 0

    for k in range(n):
        h = [0.0] * n
        h[k] = 1

        for j in range(chain_count):
            chain = [math.floor(next(generator) * 3) for _ in
                     range(chain_length + 1)]
            Q[0] = h[chain[0]] / pi[chain[0]]
            for i in range(1, chain_length + 1):
                Q[i] = Q[i - 1] * A[chain[i - 1]][chain[i]] / P[chain[i - 1]][
                    chain[i]]
            ksi[j] = sum(q * f[state] for q, state in zip(Q, chain))

        X[k] = sum(ksi) / chain_count

    return X


X_real = np.linalg.solve(np.array(g_A_real), np.array(g_f))

R = np.array([[np.linalg.norm(np.array(solve(g_A, g_f, length, count)) - X_real)
               for length in
               range(min_chain_length, max_chain_length + 1, length_step)]
              for count in
              range(min_chain_count, max_chain_count + 1, count_step)])
x, y = np.meshgrid(range(min_chain_length, max_chain_length + 1, length_step),
                   range(min_chain_count, max_chain_count + 1, count_step), )
plt.figure()
plt.title('||R||')
p = plt.pcolormesh(x, y, R, shading='nearest')
plt.colorbar(p)
plt.show()
