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
        # y=ax^b    \ ln
        # ln(y) = ln(a) + b*ln(x)

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
