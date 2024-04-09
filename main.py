import argparse
from utils import *
from math import sqrt
from scipy.interpolate import interp1d
from PIL import Image
import numpy as np
from A_matrix import A_matrix
from scipy.ndimage import gaussian_filter


def IoU(result, gt):
    im_result = np.array(Image.open(result))
    im_gt = np.array(Image.open(gt))
    if im_result.shape == im_gt.shape:
        im_result //= 255
        im_gt //= 255

        intersection = im_result * im_gt
        union = im_result + im_gt - intersection

        return np.sum(intersection) / np.sum(union)
    else:
        return None


# def get_potential_matrix(image, weight_line, weight_edge):
#     # Sobel filter kernels
#     sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
#
#     # Convolve with Sobel filters
#     grad_x = convolve(image, sobel_x)
#     grad_y = convolve(image, sobel_y)
#
#     # Calculate gradient magnitude
#     gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
#
#     # Calculate potential matrix
#     potential_matrix = weight_line * image + weight_edge * gradient_mag ** 2
#
#     return potential_matrix
#
#
# def get_energy_matrix(potential_matrix, K):
#     sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
#
#     # Convolve with Sobel filters
#     grad_x = convolve(potential_matrix, sobel_x)
#     grad_y = convolve(potential_matrix, sobel_y)
#
#     grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
#
#     normalized_grad = np.divide(grad_x, grad_mag, where=grad_mag != 0), np.divide(grad_y, grad_mag, where=grad_mag != 0)
#
#     # Calculate external energy matrix
#     ext_energy_mat = -K * np.stack(normalized_grad)
#
#     return ext_energy_mat
def criterion(img1, img2):
    if img1.shape == img2.shape:
        s = (np.sum((img1 - img2) ** 2))
        return s / (img1.shape[0] * img1.shape[1])
    else:
        return None


def get_potential_matrix(image, weight_line, weight_edge):
    kernel = np.array([-1, 0, 1])
    m = kernel.shape[0]
    temp = np.concatenate((image, image[:, -1].reshape(-1, 1)), axis=1)
    temp = np.concatenate((temp[:, 0].reshape(-1, 1), temp), axis=1)
    num_rows, num_cols = image.shape
    gradient_x = image.copy()
    gradient_y = image.copy()
    for col in range(num_cols):
        gradient_x[:, col] = temp[:, col: col + m] @ kernel
    temp = np.concatenate((image[0].reshape(1, -1), image), axis=0)
    temp = np.concatenate((temp, temp[-1].reshape(1, -1)), axis=0)
    for row in range(num_rows):
        gradient_y[row, :] = np.transpose(temp[row: row + m, :]) @ kernel

    gradient = np.stack((gradient_x, gradient_y))

    num_rows, num_cols = gradient.shape[1:]
    gradient_magnitude = np.zeros((num_rows, num_cols), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            gradient_magnitude[i, j] = np.sqrt(gradient[0, i, j] ** 2 + gradient[1, i, j] ** 2)

    potential_matrix = weight_line * image + weight_edge * gradient_magnitude ** 2
    return potential_matrix


def get_energy_matrix(potential_matrix, constant_K):
    kernel = np.array([-1, 0, 1])
    m = kernel.shape[0]
    temp = np.concatenate((potential_matrix, potential_matrix[:, -1].reshape(-1, 1)), axis=1)
    temp = np.concatenate((temp[:, 0].reshape(-1, 1), temp), axis=1)
    num_rows, num_cols = potential_matrix.shape
    gradient_x = potential_matrix.copy()
    gradient_y = potential_matrix.copy()
    for col in range(num_cols):
        gradient_x[:, col] = temp[:, col: col + m] @ kernel
    temp = np.concatenate((potential_matrix[0].reshape(1, -1), potential_matrix), axis=0)
    temp = np.concatenate((temp, temp[-1].reshape(1, -1)), axis=0)
    for row in range(num_rows):
        gradient_y[row, :] = np.transpose(temp[row: row + m, :]) @ kernel

    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    result = gradient_x.copy()
    for i in range(num_rows):
        for j in range(num_cols):
            if gradient_magnitude[i, j] != 0:
                result[i, j] /= gradient_magnitude[i, j]

    normalized_gradient = np.stack((result, gradient_y / gradient_magnitude))
    external_energy_matrix = -constant_K * normalized_gradient
    return external_energy_matrix


def get_F(f, snake):
    x_coords, y_coords = snake[:, 0], snake[:, 1]
    f_interp = np.zeros(snake.shape)

    x1, y1 = np.floor(x_coords).astype(int), np.floor(y_coords).astype(int)
    x2, y2 = np.minimum(x1 + 1, f[0].shape[1] - 1), np.minimum(y1 + 1, f[0].shape[0] - 1)

    f11, f12 = f[0][y1, x1], f[0][y2, x1]
    f21, f22 = f[0][y1, x2], f[0][y2, x2]
    dx, dy = x_coords - x1, y_coords - y1

    f_interp[:, 0] = (1 - dx) * (1 - dy) * f11 + dx * (1 - dy) * f21 + (1 - dx) * dy * f12 + dx * dy * f22

    f11, f12 = f[1][y1, x1], f[1][y2, x1]
    f21, f22 = f[1][y1, x2], f[1][y2, x2]

    f_interp[:, 1] = (1 - dx) * (1 - dy) * f11 + dx * (1 - dy) * f21 + (1 - dx) * dy * f12 + dx * dy * f22

    return f_interp


def reparametrization(snake):
    res = np.zeros(snake.shape)
    snake = np.concatenate((snake, snake[0].reshape(1, -1)), axis=0)
    dist = np.cumsum(np.sqrt(np.sum(np.diff(snake, axis=0) ** 2, axis=1)))
    dist = np.insert(dist, 0, 0)

    interpolator = interp1d(dist, snake[:, 0], kind='cubic')
    res[:, 0] = interpolator(np.linspace(0, dist[-1], res.shape[0]))

    interpolator = interp1d(dist, snake[:, 1], kind='cubic')
    res[:, 1] = interpolator(np.linspace(0, dist[-1], res.shape[0]))

    return res


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', type=str)
    parser.add_argument('initial_snake', type=str)
    parser.add_argument('output_image', type=str)
    parser.add_argument('alpha', type=float)
    parser.add_argument('beta', type=float)
    parser.add_argument('tau', type=float)
    parser.add_argument('w_line', type=float)
    parser.add_argument('w_edge', type=float)
    parser.add_argument('kappa', type=float)

    return parser.parse_args()


def load_initial_snake(initial_snake_path):
    initial_snake = np.loadtxt(initial_snake_path)[: -1]
    return initial_snake


def calculate_inverse_matrix(initial_snake, alpha, betta, tau):
    int_energy_mat = A_matrix(initial_snake.shape[0], alpha, betta)
    N = int_energy_mat.shape[0]
    inverse_mat = np.linalg.inv(np.identity(N) + int_energy_mat * tau)
    return inverse_mat


def preprocess_image(input_image_path, sigma):
    img = np.array(Image.open(input_image_path), dtype="float64")
    filtered_img = gaussian_filter(img, sigma)
    return filtered_img


def optimize_snake(initial_snake, inverse_mat, tau, ext_energy_mat, kappa, epsilon):
    prev_snake = initial_snake.copy()
    while True:
        new_snake = update_snake(prev_snake, inverse_mat, tau, ext_energy_mat, kappa)
        if criterion(new_snake, prev_snake) <= epsilon:
            break
        prev_snake = reparametrization(new_snake)
    return new_snake


def update_snake(prev_snake, inverse_mat, tau, ext_energy_mat, kappa):
    num = 4
    xt = prev_snake[:, 0]
    yt = prev_snake[:, 1]
    n = xt.shape[0]

    a = np.arange(0, n) + num
    a[a > n - 1] -= n
    b = np.arange(0, n) - num
    b[b < 0] += n

    dx = xt[a] - xt[b]
    dy = yt[a] - yt[b]

    contour_normals = np.zeros(prev_snake.shape)
    for i in range(n):
        length = sqrt(dx[0] ** 2 + dy[0] ** 2)
        contour_normals[i, 0] = -dy[i] / length
        contour_normals[i, 1] = dx[i] / length

    # (I + tau * A) X_t+1 = X_t + tau * F(X_t)
    new_snake = inverse_mat @ (prev_snake + tau * (get_F(ext_energy_mat, prev_snake) + kappa * contour_normals))
    return new_snake


def save_segmented_image(output_image_path, segmented_image, original_image):
    save_mask(output_image_path, segmented_image, original_image)


def evaluate_segmentation(output_image_path, ground_truth_mask_path):
    iou_score = IoU(output_image_path, ground_truth_mask_path)
    print('IoU: ', iou_score)


def snake_segmentation(args):
    print(args)
    initial_snake = load_initial_snake(args.initial_snake)
    inverse_mat = calculate_inverse_matrix(initial_snake, args.alpha, args.beta, args.tau)
    sigma = 1
    filtered_img = preprocess_image(args.input_image, sigma)
    potential_matrix = get_potential_matrix(filtered_img, args.w_line, args.w_edge)
    ext_energy_mat = get_energy_matrix(potential_matrix, 0.9)
    segmented_snake = optimize_snake(initial_snake, inverse_mat, args.tau, ext_energy_mat, args.kappa, 0.007)
    img = np.array(Image.open(args.input_image), dtype="float64")
    save_segmented_image(args.output_image, segmented_snake, img)

    iou_score = IoU(args.output_image, 'data/coffee_mask.png')
    print('IoU: ', iou_score)


def main():
    args = parse_arguments()
    snake_segmentation(args)


# {'alpha': 3.249922516582142e-05, 'beta': 3.8800004912090396e-06, 'kappa': 0.6292380864134605, 'tau': 0.25670589655940373}5
# python main.py data\astranaut.png data\astranaut_init_snake.txt astranaut_result.png 5.170934527338438e-07 1.5021869500259493e-08 0.5357580464005591 -1 -1 0.23965087353823147
# python main.py data\coffee.png data\coffee_init_snake.txt coffee_result.png 0.000011236  0.000000123   0.9876812  -0.7   -0.6  0.012316
# python main.py data\coins.png data\coins_init_snake.txt coins_result.png 1.527149262989133e-07 6.050736015114122e-06 0.8738369354330151 -1 -1 0.013092498577924345
# python main.py data\microarray.png data\microarray_init_snake.txt microarray_result.png 0.00000101 0.00000000112 0.512  -0.6   -1.5 -0.31231234
# python main.py data\nucleus.png data\nucleus_init_snake.txt nucleus_result.png 0.00005576  0.0000003234   1.123123  -0.9123321   -1.1123123  -0.222345734132
if __name__ == "__main__":
    main()
