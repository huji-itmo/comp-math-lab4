import numpy as np

from helpers.interpretationR import interpretR
from helpers.interpretationCorrel import interCorrel

NAME = 3

def linealApprox(x,y) -> dict[str,float]:
    print("")
    print("--- Линейная ---")
    n = len(x)
    sumX = sum(x)
    sumY = sum(y)
    sumX2 = sum(X**2 for X in x)
    sumXY = sum(X * Y for X, Y in zip(x, y))
    A = np.array([[sumX2, sumX], [sumX, n]])
    B = np.array([sumXY, sumY])
    try:
        solution = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        print("Система уравнений вырождена, решение не существует")
        return None
    if solution is not None:
        a0, a1 = solution
    else:
        print("ошибка в вычислении матрицы")

    def polinomModel(x):
        return a0*x + a1
    
    fi = []
    ei = []
    S = 0
    fiAverage = 0
    for i in range(n):
        fi.append(polinomModel(x[i]))
        ei.append(fi[i] - y[i])
        S += ei[i] ** 2
        fiAverage += fi[i]
    delta = np.sqrt(S / n)
    fiAverage = 1 / n * sum(fi)

    ss_total = sum((yi - fiAverage)**2 for yi in y)
    
    R2 = 1 - (S / ss_total)
    x_mean = sumX / n
    y_mean = sumY / n
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = np.sqrt(sum((xi - x_mean)**2 for xi in x) * sum((yi - y_mean)**2 for yi in y))
    r = numerator / denominator

    print(f"Формула: y = {a0:.6f}x + {a1:.6f}")
    print(f"Мера отклонения: S = {S:.6f}")
    print(f"Среднеквадратичное отклонение: 𝜹 = {delta:.6f}")
    print(f"Достоверность аппроксимации: R² = {R2:.6f}")
    print(f"Коэффицент корреляции: r = {r:.6f}")

    interpretR(R2)
    interCorrel(r)
    return {
        "a0": a0,
        "a1": a1,
        "S": S,
        "delta": round(delta,10),
        "R2": R2,
        "r": r,
        "name": NAME,
        "model": polinomModel
    }