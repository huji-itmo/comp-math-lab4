from typing import Callable, Tuple
import numpy as np

from helpers.interpretationR import interpretR
from helpers.interpretationCorrel import interCorrel
from least_squares_method import least_squares_method


def get_approximation_characteristics(
    x: np.ndarray,
    y: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
) -> dict[str, float] | None:

    n = x.size

    f = func(x)
    S = np.sum(np.pow(f - y, 2))
    delta = np.sqrt(S / n)
    f_mean = np.mean(f)
    R2 = 1 - np.sum(np.pow(y - f, 2)) / np.sum(np.pow(y - f_mean, 2))

    print(f"Мера отклонения: S = {S:.6f}")
    print(f"Среднеквадратичное отклонение: 𝜹 = {delta:.6f}")
    print(f"Достоверность аппроксимации: R² = {R2:.6f}")

    interpretR(R2)
    return {
        "S": S,
        "delta": delta,
        "R2": R2,
    }
