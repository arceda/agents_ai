
import matplotlib.pyplot as plt
import numpy as np

def plot_quadratic(a, b, c, x_range):
    """
    Plot a quadratic function of the form f(x) = ax^2 + bx + c.

    Parameters:
    a, b, c (float): Coefficients of the quadratic equation
    x_range (tuple): Range of x values as (xmin, xmax)
    """
    x = np.linspace(x_range[0], x_range[1], 400)
    y = a * x**2 + b * x + c
    
    plt.figure()
    plt.plot(x, y, label=f'{a}x² + {b}x + {c}')
    plt.title('Quadratic Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_linear(m, b, x_range):
    """
    Plot a linear function of the form f(x) = mx + b.

    Parameters:
    m, b (float): Slope and y-intercept of the linear equation
    x_range (tuple): Range of x values as (xmin, xmax)
    """
    x = np.linspace(x_range[0], x_range[1], 400)
    y = m * x + b
    
    plt.figure()
    plt.plot(x, y, label=f'{m}x + {b}')
    plt.title('Linear Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()
