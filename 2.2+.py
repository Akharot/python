import matplotlib.pyplot as plt
import numpy as np

def function(x):
    return x**3 - np.exp(x) + 1

def derivative(x):
    return 3 * x**2 - np.exp(x)

def grid(A, N, h, list_x):
    for i in range(N + 1):
        list_x.append(A + h * i)

def a (N, list_x, list_a):
    for k in range(N):
        list_a.append(function(list_x[k + 1]))

def val_b(N, list_x, A, B, list_value_b):
    for k in range(1, N):
        x_k = list_x[k + 1]
        x_k_2 = list_x[k - 1]
        list_value_b.append(3 * (function(x_k) - function(x_k_2)) / h)
    list_value_b[0] -= derivative(A)
    list_value_b[N - 2] -= derivative(B)

def thomas_algorithm(d, x, b_n):
    a = []
    b = []
    c = []

    for i in range(N - 1):
        a.append(1.0)
        b.append(4.0)
        c.append(1.0)

    # Прямой ход метода прогонки
    for i in range(1, N - 1):
        m = a[i] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    # Обратный ход метода прогонки
    for i in range(N):
        x.append(0.0)

    x[N - 2] = d[N - 2] / b[N - 2]
    for i in range(N - 3, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    x[N - 1] = b_n

def c(list_x, list_c, N ,list_b):
    list_c.append((2 * list_b[0] + b_0 - 3 * (function(list_x[1]) - function(list_x[0])) / h) / h)
    for i in range(1, N):
        list_c.append((2 * list_b[i] + list_b[i - 1] - 3 * (function(list_x[i + 1]) - function(list_x[i])) / h) / h)

def d(list_x, list_d, N ,list_b):
    list_d.append((list_b[0] + b_0 - 2 * (function(list_x[1]) - function(list_x[0])) / h) / (h ** 2))
    for i in range(1, N):
        list_d.append((list_b[i] + list_b[i - 1] - 2 * (function(list_x[i + 1]) - function(list_x[i])) / h) / (h ** 2))


A = -2
B = 4
N = 4
h = (B - A) / N
b_0 = derivative(A)
b_n = derivative(B)
list_x = []
list_a = []
list_value_b = []
list_b = []
list_c = []
list_d = []

grid(A, N, h, list_x)
a(N, list_x, list_a)
val_b(N, list_x, A, B, list_value_b)
thomas_algorithm(list_value_b, list_b, b_n)
c(list_x, list_c, N ,list_b)
d(list_x, list_d, N ,list_b)

for i in range(N):
    print(f"{list_a[i]:.5f} + {list_b[i]:.5f}(x - {list_x[i + 1]:.5f}) + {list_c[i]:.5f}(x - {list_x[i + 1]:.5f})^2 + {list_d[i]:.5f}(x - {list_x[i + 1]:.5f})^3")


# Построение графиков
plt.figure(figsize=(14, 10))

# Подграфик 1: Сплайны и исходная функция
plt.subplot(2, 2, 1)
x_vals = np.linspace(A, B, 1000)
y_vals = np.zeros_like(x_vals)

# Рисуем исходную функцию
y_function = np.array([function(x) for x in x_vals])
plt.plot(x_vals, y_function, label="f(x) = x^3 - e^x + 1", color="blue", linestyle="--")

# Рисуем сплайны
for i in range(N):
    x_start = list_x[i]
    x_end = list_x[i + 1]
    mask = (x_vals >= x_start) & (x_vals <= x_end)
    x_segment = x_vals[mask]
    y_segment = (
        list_a[i] +
        list_b[i] * (x_segment - list_x[i + 1]) +
        list_c[i] * (x_segment - list_x[i + 1])**2 +
        list_d[i] * (x_segment - list_x[i + 1])**3
    )
    y_vals[mask] = y_segment
    plt.plot(x_segment, y_segment, label=f"Spline {i + 1}")

# Рисуем исходные точки
plt.scatter(list_x, [function(x) for x in list_x], color='red', label='Исходные точки')

plt.title("Кубические сплайны и исходная функция")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# Подграфик 2: Зависимость фактической ошибки на отрезке
plt.subplot(2, 2, 2)

errors = np.zeros_like(x_vals)

# Вычисляем ошибку для каждого интервала
for i in range(N):
    x_start = list_x[i]
    x_end = list_x[i + 1]
    mask = (x_vals >= x_start) & (x_vals <= x_end)
    x_segment = x_vals[mask]

    y_segment = (
        list_a[i] +
        list_b[i] * (x_segment - list_x[i + 1]) +
        list_c[i] * (x_segment - list_x[i + 1])**2 +
        list_d[i] * (x_segment - list_x[i + 1])**3
    )

    f_original = function(x_segment)
    errors[mask] = f_original - y_segment


plt.plot(x_vals, errors, label="Фактическая ошибка", color="green")
plt.title("Зависимость фактической ошибки на отрезке")
plt.xlabel("x")
plt.ylabel("Ошибка")
plt.legend()
plt.grid(True)


# Подграфик 3: Зависимость максимальной ошибки от числа узлов
plt.subplot(2, 2, 3)


max_errors = []

N_values = range(4, 100)


for N in N_values:
    h = (B - A) / N
    list_x = []
    list_a = []
    list_value_b = []
    list_b = []
    list_c = []
    list_d = []


    grid(A, N, h, list_x)
    a(N, list_x, list_a)
    val_b(N, list_x, A, B, list_value_b)
    thomas_algorithm(list_value_b, list_b, b_n)
    c(list_x, list_c, N, list_b)
    d(list_x, list_d, N, list_b)

    # Вычисление ошибки в средних точках
    errors = []
    for i in range(N):
        x_mid = (list_x[i] + list_x[i + 1]) / 2  # Средняя точка между узлами
        y_spline = (
            list_a[i] +
            list_b[i] * (x_mid - list_x[i + 1]) +
            list_c[i] * (x_mid - list_x[i + 1])**2 +
            list_d[i] * (x_mid - list_x[i + 1])**3
        )
        y_function = function(x_mid)
        errors.append(abs(y_function - y_spline))

    max_errors.append(max(errors))


plt.plot(N_values, max_errors, label="Максимальная ошибка", color="purple", marker="o")
plt.title("Зависимость максимальной ошибки от числа узлов")
plt.yscale('log')
plt.xlabel("Число узлов (N)")
plt.ylabel("Максимальная ошибка")
plt.legend()
plt.grid(True)



# Подграфик 4: Зависимость максимальной ошибки от граничных значений
plt.subplot(2, 2, 4)

N = 40

max_errors = []

boundary_values = np.linspace(11.5, 12, 50)

for b_0 in boundary_values:
    h = (B - A) / N
    list_x = []
    list_a = []
    list_value_b = []
    list_b = []
    list_c = []
    list_d = []

    grid(A, N, h, list_x)
    a(N, list_x, list_a)

    val_b(N, list_x, A, B, list_value_b)
    list_value_b[0] = list_value_b[0] + derivative(A)- b_0


    thomas_algorithm(list_value_b, list_b, b_n)
    c(list_x, list_c, N, list_b)
    d(list_x, list_d, N, list_b)

    errors = []
    for i in range(N):
        x_mid = (list_x[i] + list_x[i + 1]) / 2  # Средняя точка между узлами
        y_spline = (
            list_a[i] +
            list_b[i] * (x_mid - list_x[i + 1]) +
            list_c[i] * (x_mid - list_x[i + 1])**2 +
            list_d[i] * (x_mid - list_x[i + 1])**3
        )
        y_function = function(x_mid)
        errors.append(abs(y_function - y_spline))

    max_errors.append(max(errors))

plt.plot(boundary_values, max_errors, label="Максимальная ошибка", color="orange", marker="o")
y_fun = [-10, 0.01]
x_fun = [11.864664716763388, 11.864664716763388]
plt.plot(x_fun, y_fun, label="x = 11.864664716763388", color="blue", linestyle="--")
plt.title("Зависимость максимальной ошибки от граничных значений")
plt.xlabel("Граничное значение f'(A)")
plt.yscale('log')

plt.ylabel("Максимальная ошибка")
plt.legend()
plt.grid(True)





plt.tight_layout()
plt.show()