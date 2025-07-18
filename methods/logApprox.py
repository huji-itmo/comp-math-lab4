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
        # y=a*ln(x) + b

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
