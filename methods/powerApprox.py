from typing import Callable, Tuple
import numpy as np

from approximation_characteristics import get_approximation_characteristics
from helpers.interpretationR import interpretR
from least_squares_method import least_squares_method


def powerApprox(
    x: np.ndarray, y: np.ndarray
) -> Tuple[str, dict[str, float], Callable[[np.ndarray], np.ndarray]] | None:
    print("")
    print("--- Степенная ---")

    if np.any(x <= 0) or np.any(y <= 0):
        print(
            "Ошибка: x или y содержит неположительные значения. Логарифмирование невозможно."
        )
        return None

    try:
        solution = least_squares_method(np.log(x), np.log(y), 1)
    except np.linalg.LinAlgError:
        print("Система уравнений вырождена, решение не существует")
        return None
    if solution is None:
        print("ошибка в вычислении матрицы")
        return None

    def solution_to_string():
        ln_a, b = solution

        a = np.exp(ln_a)

        return f"y = {a:.6f} * x^{b:.6f}"

    def func(x: np.ndarray):
        ln_a, b = solution

        a = np.exp(ln_a)

        return a * np.pow(x, b)

    print(f"Формула: {solution_to_string()}")
    res = get_approximation_characteristics(x, y, func)

    if res is not None:
        return ("Степенная", res, func)
    else:
        return None

    # n = len(x)

    # if any(val <= 0 for val in x) or any(val <= 0 for val in y):
    #     print("Метод неприменим. Все значения x и y должны быть положительными.")
    #     return None

    # lnx = np.log(x)
    # lny = np.log(y)

    # sumlnx = lnx.sum()
    # sumlny = lny.sum()
    # sumlnx2 = (lnx**2).sum()
    # sumlnxlny = (lnx * lny).sum()

    # A = np.array([[sumlnx2, sumlnx], [sumlnx, n]])
    # B = np.array([sumlnxlny, sumlny])

    # try:
    #     solution = np.linalg.solve(A, B)
    # except np.linalg.LinAlgError:
    #     print("Система уравнений вырождена")
    #     return None

    # b, ln_a = solution
    # a = np.exp(ln_a)

    # def polinomModel(x_val):
    #     return a * x_val**b

    # fi = polinomModel(np.array(x))
    # ei = y - fi
    # S = (ei**2).sum()
    # delta = np.sqrt(S / n)

    # y_mean = np.mean(y)
    # ss_total = ((y - y_mean) ** 2).sum()
    # R2 = 1 - (S / ss_total)

    # print(f"Формула: y = {a:.6f} * x^{b:.6f}")
    # print(f"Мера отклонения: S = {S:.6f}")
    # print(f"Среднеквадратичное отклонение: δ = {delta:.6f}")
    # print(f"Достоверность аппроксимации: R² = {R2:.6f}")

    # interpretR(R2)

    # return {
    #     "a": a,
    #     "b": b,
    #     "S": S,
    #     "delta": round(delta, 10),
    #     "R2": R2,
    #     "name": NAME,
    #     "model": polinomModel,
    # }
