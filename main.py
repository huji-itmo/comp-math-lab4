from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from methods.cubeApprox import cubeApprox
from methods.exponentialApprox import exponentialApprox
from methods.linearApprox import linealApprox
from methods.squareApprox import squareApprox
from methods.powerApprox import powerApprox
from methods.logApprox import logApprox

names = {
    1: "Кубическая",
    2: "Экспоненциальная",
    3: "Линейная",
    4: "Логарифмическая",
    5: "Степенная",
    6: "Квадратичная"
}

def readData():
    with open("input.txt", "r") as file:
        x = np.array(list(map(float, file.readline().split())))
        y = np.array(list(map(float, file.readline().split())))
    return x, y

def drawFunctions(x, y, results: List[Dict]):

    plt.figure(figsize=(12, 8))
    
    plt.scatter(x, y, c='red', s=70, label='Исходные данные', zorder=3)
    
    x_min = min(x)
    x_max = max(x)
    margin = 0.1 * (x_max - x_min) 
    x_smooth = np.linspace(x_min - margin, x_max + margin, 500)
    
    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink']
    
    for i, res in enumerate(results):
        if not res: 
            continue
            
        model = res['model']
        y_smooth = model(x_smooth)
        
        plt.plot(
            x_smooth, 
            y_smooth, 
            color=colors[i],
            linewidth=2,
            label=f"{names.get(res['name'])} (R²={res['R2']:.4f})"
        )
    
    plt.title('Сравнение методов аппроксимации', fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    plt.savefig('approximations_comparison.png', dpi=300)

def countResult(arr):
    print("\n--- Подсчёт результатов ---")
    
    minDelta = float('inf')
    for func in arr:
        if func.get('delta') < minDelta:
            minDelta = func.get('delta')
    
    bestModels = []
    for func in arr:
        if func.get('delta') == minDelta:
            bestModels.append(func)
    
    print(f"Минимальное СКО: {minDelta}")
    print("Лучшие модели:")
    if len(bestModels) != 0:
        for model in bestModels:
            print(f"- {names.get(model.get('name'))} (СКО: {model.get('delta')})")
    else:
        print("Решения не найдено")


    
def main():
    x,y = readData()
    if len(x) != len(y):
        print("ошибка ввода точек")
        return
    print("--- Вычисления ---")
    
    results = [
        squareApprox(x, y),
        linealApprox(x,y),
        cubeApprox(x,y),
        powerApprox(x,y),
        exponentialApprox(x,y),
        logApprox(x,y)
    ]

    valid_results = [res for res in results if res]
    
    countResult(valid_results)
    drawFunctions(x, y, valid_results)

if __name__ == "__main__":
    main()
