import numpy as np


import numpy as np


def least_squares_method(X: np.ndarray, Y: np.ndarray, degree: int) -> np.ndarray:
    """
    Perform least squares regression to fit a polynomial of degree
    to the data points (X, Y).
    """
    assert X.ndim == 1 and Y.ndim == 1, "X and Y must be 1D arrays"
    assert X.size == Y.size, "X and Y must have the same number of elements"
    assert degree >= 0, "Degree must be positive"

    n = X.size

    def get_x_nth_degree_sum(_degree: int) -> float:
        assert _degree >= 0
        return np.sum(np.power(X, _degree))

    matrix = []
    for i in range(degree + 1):
        row = []
        for j in range(degree + 1):
            row.append(get_x_nth_degree_sum(i + j))
        matrix.append(row)

    matrix = np.array(matrix)

    def get_x_nth_degree_mul_Y_sum(_degree: int) -> float:
        assert _degree >= 0
        return np.sum(np.power(X, _degree) * Y)

    answers = []
    for i in range(degree + 1):
        answers.append(get_x_nth_degree_mul_Y_sum(i))
    answers = np.array(answers)

    return np.linalg.solve(matrix, answers)
