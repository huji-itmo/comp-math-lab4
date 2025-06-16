from typing import Callable, Tuple
import numpy as np

from approximation_characteristics import get_approximation_characteristics
from helpers.interpretationR import interpretR
from least_squares_method import least_squares_method


def fifthApprox(
    x: np.ndarray, y: np.ndarray
) -> Tuple[str, dict[str, float], Callable[[np.ndarray], np.ndarray]] | None:
    print("")
    print("--- Пятая степень ---")

    try:
        solution = least_squares_method(x, y, 5)
    except np.linalg.LinAlgError:
        print("Система уравнений вырождена, решение не существует")
        return None
    if solution is None:
        print("ошибка в вычислении матрицы")
        return None

    def solution_to_string():
        a0, a1, a2, a3, a4, a5 = solution
        return f"y = {a5:.6f}x^5 + {a4:.6f}x^4 + {a3:.6f}x³ + {a2:.6f}x² + {a1:.6f}x + {a0:.6f}"

    def func(x: np.ndarray):
        a0, a1, a2, a3, a4, a5 = solution
        return a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4 + a5 * x**5

    print(f"Формула: {solution_to_string()}")
    res = get_approximation_characteristics(x, y, func)
    if res is not None:
        return ("Пятая степень", res, func)
    else:
        return None
