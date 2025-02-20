import numpy as np
import matplotlib.pyplot as plt
import math

def original_function(x):
    return x ** 3 - np.exp(x) + 1

def divided_differences(list_x, list_y):
    N = len(list_x) - 1
    dd = np.zeros((N + 1, N + 1))
    dd[:, 0] = list_y

    for j in range(1, N + 1):
        for i in range(N + 1 - j):
            dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / (list_x[i + j] - list_x[i])

    return dd

def newton_backward_polynomial(x, list_x, dd):
    N = len(list_x) - 1
    result = dd[N, 0]
    for j in range(1, N + 1):
        term = dd[N - j, j]
        for i in range(j):
            term *= (x - list_x[N - i])
        result += term
    return result

def print_divided_differences(dd):
    print("\nDivided Differences:")
    for row in dd:
        print(" | ".join(f"{val:10.5f}" for val in row if val != 0))

def print_newton_backward_polynomial(list_x, dd):
    N = len(list_x) - 1
    print(f"P(x) = {dd[N, 0]:.5f}", end="")
    for j in range(1, N + 1):
        print(f" + ({dd[N - j, j]:.5f}", end="")
        for i in range(j):
            print(f" * (x - {list_x[N - i]:.5f})", end="")
        print(")", end="")
    print()

def verify_polynomial(list_x, list_y, dd):
    print("\nTest")
    for i in range(len(list_x)):
        p_val = newton_backward_polynomial(list_x[i], list_x, dd)
        print(
            f"x[{i}] = {list_x[i]:.6f}, P(x) = {p_val:.6f}, f(x) = {list_y[i]:.6f}, Error = {abs(p_val - list_y[i]):.6e}")


A, B, N = -2, 4, 4
list_x = np.array([(A + B) / 2 + ((B - A) / 2) * np.cos(np.pi * (2 * k + 1) / (2 * (N + 1))) for k in range(N + 1)])
list_y = original_function(list_x)

dd = divided_differences(list_x, list_y)

print("Nodes:")
for k in range(N + 1):
    print(f"x[{k}] = {list_x[k]:.6f}, y[{k}] = {list_y[k]:.6f}")

print_divided_differences(dd)
print_newton_backward_polynomial(list_x, dd)
verify_polynomial(list_x, list_y, dd)

x_vals = np.linspace(A, B, 10000)
f_original = [original_function(x) for x in x_vals]
f_newton = [newton_backward_polynomial(x, list_x, dd) for x in x_vals]
error = [f_original[i] - f_newton[i] for i in range(len(x_vals))]

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x_vals, f_original, label="Оригинальная функция", linestyle='dashed')
plt.plot(x_vals, f_newton, label="Полином Ньютона назад")
plt.scatter(list_x, list_y, color='red', label="Узлы Чебышёва")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Интерполяция полиномом Ньютона назад")
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(x_vals, error, label="Фактическая ошибка")
plt.xlabel("x")
plt.ylabel("Ошибка")
plt.title("Фактическая ошибка интерполяции")
plt.legend()
plt.grid()

plt.subplot(2, 2, 3)
n_values = range(2, 25)
max_errors = []
for n in n_values:
    list_x = [(A + B) / 2 + ((B - A) / 2) * math.cos(math.pi * (2 * k + 1) / (2 * (n + 1))) for k in range(n + 1)]
    list_y = [original_function(x) for x in list_x]
    dd = divided_differences(list_x, list_y)
    midpoints = [(list_x[i] + list_x[i + 1]) / 2 for i in range(n)]
    max_error = max(abs(original_function(x) - newton_backward_polynomial(x, list_x, dd)) for x in midpoints)
    max_errors.append(max_error)

plt.plot(n_values, max_errors, marker='o', linestyle='-', color='green', label='Max Error')
plt.yscale('log')
plt.xlabel("Число узлов")
plt.ylabel("Максимальная ошибка")
plt.title("Зависимость максимальной ошибки от числа узлов")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
