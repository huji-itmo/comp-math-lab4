import numpy as np

from helpers.interpretationR import interpretR

NAME = 4
def logApprox(x, y) -> dict[str, float]:
    print("")
    print("--- Логарифмическая ---")
    if any(val <= 0 for val in x):
        print("Метод неприменим. Все значения x должны быть положительными.")
        return None
    
    n = len(x)
    x_arr = np.array(x)
    y_arr = np.array(y)

    lnx = np.log(x_arr)
    sum_lnx = np.sum(lnx)
    sum_lnx2 = np.sum(lnx**2)
    sum_y = np.sum(y_arr)
    sum_ylnx = np.sum(y_arr * lnx)

    A = np.array([[sum_lnx2, sum_lnx], [sum_lnx, n]])
    B = np.array([sum_ylnx, sum_y])

    try:
        a, b = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        print("Система уравнений вырождена")
        return {}

    def polinomModel(x):
        return a * np.log(x) + b

    fi = polinomModel(np.array(x))
    ei = y - fi
    S = (ei**2).sum()
    delta = np.sqrt(S / n)

    y_mean = np.mean(y)
    ss_total = ((y - y_mean) ** 2).sum()
    R2 = 1 - (S / ss_total)

    print(f"Формула: y = {a:.6f} * lnx + {b:.6f}")
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
