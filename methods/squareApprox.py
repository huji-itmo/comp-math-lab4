import numpy as np

from helpers.interpretationR import interpretR

NAME = 6

def squareApprox(x, y) -> dict[str,float]:
    print("")
    print("--- Квадратичная ---")
    n = len(x)
    sumX = sum(x)
    sumY = sum(y)
    sumX2 = sum(X**2 for X in x)
    sumXY = sum(X * Y for X, Y in zip(x, y))
    sumX3 = sum(X**3 for X in x)
    sumX2Y = sum(X**2 * Y for X, Y in zip(x, y))
    sumX4 = sum(X**4 for X in x)

    A = np.array([
        [n, sumX, sumX2], 
        [sumX, sumX2, sumX3], 
        [sumX2, sumX3, sumX4]
        ]
    )
    
    B = np.array([sumY, sumXY, sumX2Y])

    try:
        solution = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        print("Система уравнений вырождена, решение не существует")
        return None
    if solution is not None:
        a0, a1, a2 = solution
    else:
        print("ошибка в вычислении матрицы")

    def polinomModel(x):
        return a0 + a1 * x + a2 * x**2

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


    print(f"Формула: y = {a2:.6f}x² + {a1:.6f}x + {a0:.6f}")
    print(f"Мера отклонения: S = {S:.6f}")
    print(f"Среднеквадратичное отклонение: 𝜹 = {delta:.6f}")
    print(f"Достоверность аппроксимации: R² = {R2:.6f}")
    interpretR(R2)

    return {
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "S": S,
        "delta": round(delta,10),
        "R2": R2,
        "name": NAME,
        "model": polinomModel
    }