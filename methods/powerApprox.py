import numpy as np

from helpers.interpretationR import interpretR

NAME = 5

def powerApprox(x: list[float], y: list[float]) -> dict[str, float]:
    print("")
    print("--- Степенная ---")

    n = len(x)

    if any(val <= 0 for val in x) or any(val <= 0 for val in y):
        print("Метод неприменим. Все значения x и y должны быть положительными.")
        return None



    lnx = np.log(x)
    lny = np.log(y)

    sumlnx = lnx.sum()
    sumlny = lny.sum()
    sumlnx2 = (lnx ** 2).sum()
    sumlnxlny = (lnx * lny).sum()

    A = np.array([
        [sumlnx2, sumlnx],
        [sumlnx, n]
    ])
    B = np.array([sumlnxlny, sumlny])

    try:
        solution = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        print("Система уравнений вырождена")
        return None

    b, ln_a = solution
    a = np.exp(ln_a)

    def polinomModel(x_val):
        return a * x_val ** b

    fi = polinomModel(np.array(x))
    ei = y - fi
    S = (ei ** 2).sum()
    delta = np.sqrt(S / n)

    y_mean = np.mean(y)
    ss_total = ((y - y_mean) ** 2).sum()
    R2 = 1 - (S / ss_total)

    print(f"Формула: y = {a:.6f} * x^{b:.6f}")
    print(f"Мера отклонения: S = {S:.6f}")
    print(f"Среднеквадратичное отклонение: δ = {delta:.6f}")
    print(f"Достоверность аппроксимации: R² = {R2:.6f}")
    
    interpretR(R2)

    return {
        "a": a,
        "b": b,
        "S": S,
        "delta": round(delta,10),
        "R2": R2,
        "name": NAME,
        "model": polinomModel
    }