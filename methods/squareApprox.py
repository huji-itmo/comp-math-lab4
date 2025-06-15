from typing import Callable, Tuple
import numpy as np

from approximation_characteristics import get_approximation_characteristics
from helpers.interpretationR import interpretR
from least_squares_method import least_squares_method


def squareApprox(
    x: np.ndarray, y: np.ndarray
) -> Tuple[str, dict[str, float], Callable[[np.ndarray], np.ndarray]] | None:
    print("")
    print("--- Квадратичная ---")

    try:
        solution = least_squares_method(x, y, 2)
    except np.linalg.LinAlgError:
        print("Система уравнений вырождена, решение не существует")
        return None
    if solution is None:
        print("ошибка в вычислении матрицы")
        return None

    def solution_to_string():
        a0, a1, a2 = solution
        return f"y = {a2:.6f}x² + {a1:.6f}x + {a0:.6f}"

    def func(x: np.ndarray):
        a0, a1, a2 = solution
        return a0 + a1 * x + a2 * x**2

    print(f"Формула: {solution_to_string()}")
    res = get_approximation_characteristics(x, y, func)

    if res is not None:
        return ("Квадратичная", res, func)
    else:
        return None
