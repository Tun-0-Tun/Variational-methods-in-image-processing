import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys


def read_initial_snake(file_name):
    return np.loadtxt(file_name)


def calculate_internal_energy(snake, alpha, beta):
    snake_extended = np.vstack((snake, snake[0]))
    dx = np.diff(snake_extended, axis=0)
    elasticity_energy = np.sum(dx[:-1] ** 2, axis=1)  # Calculate for all but the last repeated point

    # Calculate second derivative for rigidity, resulting in (N-2) elements
    d2x = np.diff(snake_extended, n=2, axis=0)
    rigidity_energy = np.sum(d2x ** 2, axis=1)

    # Adjust elasticity_energy to match the size of rigidity_energy for proper addition
    # Since we're taking a second derivative, we lose two points, thus we remove from both ends for elasticity_energy
    elasticity_energy_adjusted = elasticity_energy[1:-1]  # Adjust the size by excluding the first and last elements

    internal_energy = alpha * elasticity_energy_adjusted + beta * rigidity_energy
    return internal_energy


def calculate_external_energy(image, snake, w_line, w_edge):
    smoothed_image = gaussian_filter(image, sigma=1)
    gradient_magnitude = gaussian_gradient_magnitude(smoothed_image, sigma=1)
    x, y = np.clip(snake[:, 0], 0, image.shape[1] - 1), np.clip(snake[:, 1], 0, image.shape[0] - 1)
    line_energy = smoothed_image[y.astype(int), x.astype(int)]
    edge_energy = gradient_magnitude[y.astype(int), x.astype(int)]
    line_energy = (line_energy - np.min(line_energy)) / (np.max(line_energy) - np.min(line_energy) + 1e-10)
    edge_energy = (edge_energy - np.min(edge_energy)) / (np.max(edge_energy) - np.min(edge_energy) + 1e-10)
    external_energy = -w_line * line_energy + w_edge * edge_energy
    return external_energy


def update_snake(snake, internal_energy, external_energy, tau):
    total_energy = internal_energy + external_energy
    energy_gradient = np.zeros_like(snake)
    energy_gradient[:-1] = snake[1:] - snake[:-1]
    energy_gradient[1:] += snake[:-1] - snake[1:]
    updated_snake = snake - tau * energy_gradient * total_energy[:, None]
    return updated_snake


def main(input_image_path, initial_snake_path, output_image_path, alpha, beta, tau, w_line, w_edge):
    # Load the initial snake and the image
    snake = read_initial_snake(initial_snake_path)
    image = mpimg.imread(input_image_path) if input_image_path.endswith('.png') else plt.imread(input_image_path)

    if image.ndim == 3:  # Convert RGB to grayscale if necessary
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Perform snake updates
    for _ in range(100):  # Number of iterations can be adjusted
        internal_energy = calculate_internal_energy(snake, alpha, beta)
        external_energy = calculate_external_energy(image, snake, w_line, w_edge)
        snake = update_snake(snake, internal_energy, external_energy, tau)

    # Convert the final snake to a binary mask and save
    plt.imshow(image, cmap='gray')
    plt.plot(snake[:, 0], snake[:, 1], 'r')
    plt.show()  # Display the result; you might want to save this instead


if __name__ == "__main__":
    # if len(sys.argv) < 9:
    #     print("Usage: %s input_image initial_snake output_image alpha beta tau w_line w_edge" % sys.argv[0])
    #     sys.exit(1)
    #
    # input_image_path, initial_snake_path, output_image_path, alpha, beta, tau, w_line, w_edge = sys.argv[1:9]
    # alpha, beta, tau, w_line, w_edge = map(float, [alpha, beta, tau, w_line, w_edge])

    params = {
        "input_image_path": "data\\astranaut.png",
        "initial_snake_path": "data\\astranaut_init_snake.txt",
        "output_image_path": "data\\astranaut_result.png",
        "alpha": 0.1,
        "beta": 0.01,
        "tau": 0.1,
        "w_line": 0.5,
        "w_edge": 0.5,
    }
    #

    main(**params)
