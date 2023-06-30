import numpy as np

def compute_error_for_line_given_points(b: float, w: float, points: list) -> float:

    total_error = 0
    for index in range(0, len(points)):
        x = points[index, 0]
        y = points[index, 1]

        total_error += (y - (w * x + b)) ** 2
    
    return total_error / float(len(points))

def step_gradient(b_current: float, w_current: float, points: list, learning_rate: float) -> list:

    b_gradient = 0
    w_gradient = 0
    N= float(len(points))
    for index in range(0, len(points)):
        x = points[index, 0]
        y = points[index, 1]
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_w = w_current - (learning_rate * w_gradient)

    return [new_b, new_w]

def gradient_descent_runner(points: list, starting_b: float, starting_w: float, learning_rate: float, num_iterations: int) -> list:

    b = starting_b
    w = starting_w

    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    
    return [b, w]

def run() -> None:

    points = np.genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    print(f'Starting gradient descent at b = {initial_b}, w = {initial_w}, error = {compute_error_for_line_given_points(initial_b, initial_w, points)}')
    print('Running...')
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print(f'After {num_iterations} iterations b = {b}, w = {w}, error = {compute_error_for_line_given_points(b, w, points)}')


if __name__ == '__main__':
    run()
     


