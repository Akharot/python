import math
import matplotlib.pyplot as plt


def newton_method(func, dfunc, x0, tol=1e-6, max_iter=100):
    current_x = x0
    iterations = 0

    for _ in range(max_iter):
        f_value = func(current_x)
        f_prime_value = dfunc(current_x)

        if abs(f_value) < tol:
            break

        if f_prime_value == 0:
            raise ValueError("Производная равна нулю, метод Ньютона не применим.")

        current_x = current_x - f_value / f_prime_value
        iterations += 1

    return current_x, iterations


def bisection_method(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) >= 0:
        raise ValueError("Метод половинного деления не применим: неверные границы отрезка.")

    iterations = 0
    while (b - a) / 2 > tol and iterations < max_iter:
        c = (a + b) / 2
        if func(c) == 0:
            return c, iterations
        elif func(a) * func(c) < 0:
            b = c
        else:
            a = c
        iterations += 1

    return (a + b) / 2, iterations


# Функция и её производная для уравнения x + cos(x) = 0
func = lambda x: x + math.cos(x)
dfunc = lambda x: 1 - math.sin(x)

# Точное значение корня
true_root = -0.7390851332151607

tolerances = [10 ** -i for i in range(1, 10)]
errors_newton = []
iterations_newton = []
errors_bisection = []
iterations_bisection = []

for tol in tolerances:
    root_newton, iter_count_newton = newton_method(func, dfunc, x0=-0.5, tol=tol)
    root_bisection, iter_count_bisection = bisection_method(func, a=-1, b=0, tol=tol)

    errors_newton.append(abs(true_root - root_newton))
    iterations_newton.append(iter_count_newton)
    errors_bisection.append(abs(true_root - root_bisection))
    iterations_bisection.append(iter_count_bisection)

# Отображение двух графиков одновременно
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# График фактической ошибки от заданной точности
axs[0].plot(tolerances, errors_newton, marker='o', label='Метод Ньютона')
axs[0].plot(tolerances, errors_bisection, marker='s', label='Метод половинного деления')
axs[0].plot(tolerances, tolerances, linestyle='--', label='Линия точности')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel('Заданная точность')
axs[0].set_ylabel('Фактическая ошибка')
axs[0].legend()
axs[0].set_title('Фактическая ошибка от заданной точности')
axs[0].grid()

# График числа итераций от заданной точности
axs[1].plot(tolerances, iterations_newton, marker='o', label='Метод Ньютона')
axs[1].plot(tolerances, iterations_bisection, marker='s', label='Метод половинного деления')
axs[1].set_xscale('log')
axs[1].set_xlabel('Заданная точность')
axs[1].set_ylabel('Число итераций')
axs[1].legend()
axs[1].set_title('Число итераций от заданной точности')
axs[1].grid()

plt.tight_layout()
plt.show()
