from typing import Callable, Tuple
import numpy as np

from approximation_characteristics import get_approximation_characteristics
from helpers.interpretationR import interpretR
from least_squares_method import least_squares_method


def exponentialApprox(
    x: np.ndarray, y: np.ndarray
) -> Tuple[str, dict[str, float], Callable[[np.ndarray], np.ndarray]] | None:
    print("")
    print("--- Экспоненциальная ---")

    if np.any(y <= 0):
        print(
            "Ошибка: y содержит неположительные значения. Логарифмирование невозможно."
        )
        return None

    try:
        solution = least_squares_method(x, np.log(y), 1)
    except np.linalg.LinAlgError:
        print("Система уравнений вырождена, решение не существует")
        return None
    if solution is None:
        print("ошибка в вычислении матрицы")
        return None

    def solution_to_string():
        # y=ae^{bx}     \ ln
        # ln(y)=ln(a) + bx

        ln_a, b = solution
        a = np.exp(ln_a)

        return f"y = {a:.6f} * e^{b:.6f}x"

    def func(x: np.ndarray):
        a0, a1 = solution
        a0 = np.exp(a0)
        return a0 * np.exp(a1 * x)

    print(f"Формула: {solution_to_string()}")
    res = get_approximation_characteristics(x, y, func)

    if res is not None:
        return ("Экспоненциальная", res, func)
    else:
        return None
