import numpy as np

def create_a(n):
    # Создание единичной матрицы с помощью np.eye
    eye_matrix = np.eye(n)

    # Создание матрицы, сдвинутой на одну позицию вправо и вниз с помощью np.roll
    shifted_eye_matrix = np.roll(eye_matrix, -1, axis=1)
    shifted_eye_matrix_column = np.roll(eye_matrix, -1, axis=0)

    # Создание окончательной матрицы путем вычитания и умножения
    result_matrix = shifted_eye_matrix + shifted_eye_matrix_column - 2 * eye_matrix

    return result_matrix

def create_b(n):
    # Создание единичной матрицы с помощью np.eye
    eye_matrix = np.eye(n)

    # Создание матрицы, сдвинутой на две позиции влево и вверх с помощью np.roll
    shifted_eye_matrix_left = np.roll(eye_matrix, -2, axis=1)
    shifted_eye_matrix_up = np.roll(eye_matrix, -2, axis=0)

    # Создание матрицы, сдвинутой на одну позицию влево и вверх с помощью np.roll
    shifted_eye_matrix_left_one = np.roll(eye_matrix, -1, axis=1)
    shifted_eye_matrix_up_one = np.roll(eye_matrix, -1, axis=0)

    # Создание окончательной матрицы путем вычитания и умножения
    result_matrix = shifted_eye_matrix_left - 4 * shifted_eye_matrix_left_one + 6 * eye_matrix - \
                    4 * shifted_eye_matrix_up_one + shifted_eye_matrix_up

    return result_matrix
def A_matrix(n, alpha, beta):
    """
    Генерация матрицы по заданным параметрам alpha, beta и tau.
    """
    A = -alpha * create_a(n) * (n * n) + beta * create_b(n) * (n ** 4)
    return A



