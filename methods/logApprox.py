from typing import Callable, Tuple
import numpy as np

from approximation_characteristics import get_approximation_characteristics
from least_squares_method import least_squares_method


def logApprox(
    x: np.ndarray, y: np.ndarray
) -> Tuple[str, dict[str, float], Callable[[np.ndarray], np.ndarray]] | None:
    print("")
    print("--- Логарифмическая ---")

    if np.any(x <= 0):
        print(
            "Ошибка: x содержит неположительные значения. Логарифмирование невозможно."
        )
        return None

    try:
        solution = least_squares_method(np.log(x), y, 1)
    except np.linalg.LinAlgError:
        print("Система уравнений вырождена, решение не существует")
        return None
    if solution is None:
        print("ошибка в вычислении матрицы")
        return None

    def solution_to_string():
        b, a = solution

        return f"y = {a:.6f} * lnx + {b:.6f}"

    def func(x: np.ndarray):
        b, a = solution

        return a * np.log(x) + b

    print(f"Формула: {solution_to_string()}")
    res = get_approximation_characteristics(x, y, func)

    if res is not None:
        return ("Логарифмическая", res, func)
    else:
        return None

    # if any(val <= 0 for val in x):
    #     print("Метод неприменим. Все значения x должны быть положительными.")
    #     return None

    # n = len(x)
    # x_arr = np.array(x)
    # y_arr = np.array(y)

    # lnx = np.log(x_arr)
    # sum_lnx = np.sum(lnx)
    # sum_lnx2 = np.sum(lnx**2)
    # sum_y = np.sum(y_arr)
    # sum_ylnx = np.sum(y_arr * lnx)

    # A = np.array([[sum_lnx2, sum_lnx], [sum_lnx, n]])
    # B = np.array([sum_ylnx, sum_y])

    # try:
    #     a, b = np.linalg.solve(A, B)
    # except np.linalg.LinAlgError:
    #     print("Система уравнений вырождена")
    #     return {}

    # def polinomModel(x):
    #     return a * np.log(x) + b

    # fi = polinomModel(np.array(x))
    # ei = y - fi
    # S = (ei**2).sum()
    # delta = np.sqrt(S / n)

    # y_mean = np.mean(y)
    # ss_total = ((y - y_mean) ** 2).sum()
    # R2 = 1 - (S / ss_total)

    # print(f"Формула: y = {a:.6f} * lnx + {b:.6f}")
    # print(f"Мера отклонения: S = {S:.6f}")
    # print(f"Среднеквадратичное отклонение: δ = {delta:.6f}")
    # print(f"Достоверность аппроксимации: R² = {R2:.6f}")

    # interpretR(R2)
    # return {
    #     "a": a,
    #     "b": b,
    #     "S": S,
    #     "delta": round(delta,10),
    #     "R2": R2,
    #     "name": NAME,
    #     "model": polinomModel
    # }
