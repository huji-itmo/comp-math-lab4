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

    # n = len(x)
    # sumX = sum(x)
    # sumY = sum(y)
    # sumX2 = sum(X**2 for X in x)
    # sumXY = sum(X * Y for X, Y in zip(x, y))
    # A = np.array([[sumX2, sumX], [sumX, n]])
    # B = np.array([sumXY, sumY])
    # try:
    #     solution = np.linalg.solve(A, B)
    # except np.linalg.LinAlgError:
    #     print("Система уравнений вырождена, решение не существует")
    #     return None
    # if solution is not None:
    #     a0, a1 = solution
    # else:
    #     print("ошибка в вычислении матрицы")

    # def polinomModel(x):
    #     return a0 * x + a1

    # fi = []
    # ei = []
    # S = 0
    # fiAverage = 0
    # for i in range(n):
    #     fi.append(polinomModel(x[i]))
    #     ei.append(fi[i] - y[i])
    #     S += ei[i] ** 2
    #     fiAverage += fi[i]
    # delta = np.sqrt(S / n)
    # fiAverage = 1 / n * sum(fi)

    # ss_total = sum((yi - fiAverage) ** 2 for yi in y)

    # R2 = 1 - (S / ss_total)
    # x_mean = sumX / n
    # y_mean = sumY / n
    # numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    # denominator = np.sqrt(
    #     sum((xi - x_mean) ** 2 for xi in x) * sum((yi - y_mean) ** 2 for yi in y)
    # )
    # r = numerator / denominator

    # print(f"Формула: y = {a0:.6f}x + {a1:.6f}")
    # print(f"Мера отклонения: S = {S:.6f}")
    # print(f"Среднеквадратичное отклонение: 𝜹 = {delta:.6f}")
    # print(f"Достоверность аппроксимации: R² = {R2:.6f}")
    # print(f"Коэффицент корреляции: r = {r:.6f}")

    # interpretR(R2)
    # interCorrel(r)
    # return {
    #     "a0": a0,
    #     "a1": a1,
    #     "S": S,
    #     "delta": round(delta, 10),
    #     "R2": R2,
    #     "r": r,
    #     "name": NAME,
    #     "model": polinomModel,
    # }


def get_correlation_coefficient(x: np.ndarray, y: np.ndarray):
    x_mean = np.average(x)
    y_mean = np.average(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum(np.pow(x - x_mean, 2)) * np.sum(np.pow(y - y_mean, 2)))

    return numerator / denominator
