from typing import Callable, Tuple
import numpy as np

from approximation_characteristics import get_approximation_characteristics
from helpers.interpretationR import interpretR
from helpers.interpretationCorrel import interCorrel
from least_squares_method import least_squares_method


def linealApprox(
    x: np.ndarray, y: np.ndarray
) -> Tuple[str, dict[str, float], Callable[[np.ndarray], np.ndarray]] | None:
    print("")
    print("--- Линейная ---")

    try:
        solution = least_squares_method(x, y, 1)
    except np.linalg.LinAlgError:
        print("Система уравнений вырождена, решение не существует")
        return None
    if solution is None:
        print("ошибка в вычислении матрицы")
        return None

    def solution_to_string():
        a0, a1 = solution
        return f"y = {a1:.6f}x + {a0:.6f}"

    def func(x: np.ndarray):
        a0, a1 = solution
        return a0 + a1 * x

    print(f"Формула: {solution_to_string()}")
    res = get_approximation_characteristics(x, y, func)

    r = get_correlation_coefficient(x, y)
    print(f"Коэффицент корреляции: r = {r:.6f}")
    interCorrel(r)

    if res is not None:
        return ("Линейная", res, func)
    else:
        return None


def get_correlation_coefficient(x: np.ndarray, y: np.ndarray):
    x_mean = np.average(x)
    y_mean = np.average(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum(np.pow(x - x_mean, 2)) * np.sum(np.pow(y - y_mean, 2)))

    return numerator / denominator
