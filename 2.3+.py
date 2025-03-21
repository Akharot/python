import numpy as np
import matplotlib.pyplot as plt


def function(x):
    return x ** 5 - 3.5 * x ** 3 + 2.5 * x ** 2 - 7 * x - 6.4


def antiderivative(x):
    return (x ** 6) / 6 - (7 * x ** 4) / 8 + (5 * x ** 3) / 6 - (7 * x ** 2) / 2 - 6.4 * x


def exact_integral(A, B):
    return antiderivative(B) - antiderivative(A)


def three_eight(a, b):
    return ((b - a) / 8) * (function(a) + 3 * function((2 * a + b) / 3) + 3 * function((a + 2 * b) / 3) + function(b))


def three_eight_n(A, B, n):
    H = (B - A) / n
    S = sum(three_eight(A + i * H, A + (i + 1) * H) for i in range(n))
    return S


# Границы интегрирования
A, B = -2.4, -0.5
I_exact = exact_integral(A, B)

eps_values = np.logspace(-2, -10, 10)
n_values = []
power_of_two_values = []
errors = []

# График 1: Фактическая точность vs Заданная точность
for eps in eps_values:
    n = 1
    power_of_two = 0  # Начинаем с 2^0 = 1
    I1, I2 = three_eight_n(A, B, n), three_eight_n(A, B, 2 * n)
    while abs(I2 - I1) / 15 > eps:
        n *= 2
        power_of_two += 1  # Увеличиваем степень двойки
        I1, I2 = I2, three_eight_n(A, B, 2 * n)

    errors.append(abs(I2 - I_exact))
    n_values.append(n)
    power_of_two_values.append(power_of_two)  # Сохраняем степень двойки

plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
plt.loglog(eps_values, errors, 'bo-', label='Фактическая ошибка')
plt.loglog(eps_values, eps_values, 'r--', label='биссектриса')
plt.xlabel('Заданная точность ε')
plt.ylabel('Фактическая ошибка')
plt.legend()
plt.grid()
plt.title('Фактическая точность vs Заданная точность')

# График 2: Число разбиений vs Заданная точность
plt.subplot(1, 3, 2)
plt.plot(eps_values, power_of_two_values, 'go-', label='Степень двойки')
plt.xscale('log')
plt.xlabel('Заданная точность ε')
plt.ylabel('Степень двойки')
plt.legend()
plt.grid()
plt.title('Степень двойки vs Заданная точность')

# График 3: Фактическая точность vs Длина отрезка разбиения
h_values = [(B - A) / n for n in n_values]
p = 4  # Порядок метода
plt.subplot(1, 3, 3)
plt.loglog(h_values, errors, 'mo-', label='Фактическая ошибка')
plt.loglog(h_values, [h ** p for h in h_values], 'k--', label=f'h^{p}')
plt.xlabel('Длина подотрезка h')
plt.ylabel('Фактическая ошибка')
plt.legend()
plt.grid()
plt.title('Фактическая точность vs Длина разбиения')

plt.tight_layout()
plt.show()
