import numpy as np

from helpers.interpretationR import interpretR

NAME = 2


def exponentialApprox(x, y):
    print("")
    print("--- Экспоненциальная ---")

    if any(val <= 0 for val in y):
        print("Метод неприменим. Все значения y должны быть положительными.")
        return None

    n = len(x)
    x_arr = np.array(x)
    y_arr = np.array(y)
    ln_y = np.log(y_arr)

    sum_x = np.sum(x_arr)
    sum_ln_y = np.sum(ln_y)
    sum_xx = np.sum(x_arr**2)
    sum_x_ln_y = np.sum(x_arr * ln_y)

    A = np.array([[sum_xx, sum_x], [sum_x, n]])
    B = np.array([sum_x_ln_y,sum_ln_y])
    

    try:
        solution = np.linalg.solve(A, B)
        a1 = solution[0]  
        a0 = solution[1]
    except np.linalg.LinAlgError:
        print("Система уравнений вырождена")
        return {}

    a0 = np.exp(a0)

    def polinomModel(x):
        return a0 * np.exp(a1 * x)

    y_pred = polinomModel(x_arr)
    residuals = y_arr - y_pred
    S = np.sum(residuals**2)
    delta = np.sqrt(S / n)

    y_mean = np.mean(y_arr)
    total_sum_squares = np.sum((y_arr - y_mean) ** 2)
    R2 = 1 - (S / total_sum_squares)

    print(f"Формула: y = {a0:.6f} * e^{a1:.6f}x")
    print(f"Мера отклонения: S = {S:.6f}")
    print(f"Среднеквадратичное отклонение: δ = {delta:.6f}")
    print(f"Достоверность аппроксимации: R² = {R2:.6f}")

    interpretR(R2)
    
    return {"a0": a0, "a1": a1, "S": S, "delta": round(delta,10), "R2": R2, "name": NAME, "model": polinomModel}
