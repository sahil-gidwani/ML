import numpy as np
import matplotlib.pyplot as plt

# Define a simple quadratic function
def f(x):
    return (x + 3) ** 2

# Define the derivative of the quadratic function
def df(x):
    return 2 * (x + 3)

# Implement the gradient descent algorithm
def gradient_descent(initial_x, learning_rate, num_iterations):
    x = initial_x
    x_history = [x]

    # Perform a specified number of iterations
    for i in range(num_iterations):
        # Calculate the gradient of the function at the current point
        gradient = df(x)
        
        # Update the current point using the gradient and the learning rate
        x = x - learning_rate * gradient

        # Store the updated point in the history for visualization
        x_history.append(x)

    return x, x_history

# Define the initial point, learning rate, and number of iterations
initial_x = 2
learning_rate = 0.1
num_iterations = 50

# Apply gradient descent to find the local minimum
x, x_history = gradient_descent(initial_x, learning_rate, num_iterations)

# Print the local minimum found by the algorithm
print("Local minimum: {:.2f}".format(x))

# Create a range of x values to plot
x_vals = np.linspace(-10, 4, 100)

# Plot the function f(x)
plt.plot(x_vals, f(x_vals))

# Plot the values of x at each iteration as red 'x' markers
plt.plot(x_history, f(np.array(x_history)), 'rx')

# Label the axes and add a title to the plot
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent')

# Show the plot
plt.show()

""""
Gradient Descent is an optimization algorithm used to minimize a cost function (or loss function) in the context of machine learning and deep learning. It is one of the most fundamental and widely used techniques for training models, such as linear regression, logistic regression, and neural networks. The primary goal of Gradient Descent is to find the minimum of a cost function by iteratively updating model parameters. Here's an explanation of how it works:

1. **Cost Function**: Gradient Descent begins with a cost function, often denoted as J(θ), where θ represents the model parameters (weights and biases). The cost function quantifies how well the model's predictions match the actual target values. The objective is to minimize this cost.

2. **Initialization**: Initially, you need to choose an initial set of parameters θ. This is typically done randomly or with small values.

3. **Gradient Calculation**: The next step is to compute the gradient of the cost function with respect to each parameter in θ. The gradient represents the direction and magnitude of the steepest increase in the cost function. It points to where the cost function is increasing the fastest.

4. **Parameter Update**: In this step, you adjust the model parameters θ by moving them in the opposite direction of the gradient. The idea is to move the parameters in such a way that the cost function decreases. The update rule is typically defined as:
   
   θ = θ - learning_rate * gradient

   Here, the learning rate (often denoted as α) controls the step size in the parameter space. It's a hyperparameter that you need to tune. A smaller learning rate leads to more precise updates but slower convergence, while a larger learning rate can speed up convergence but may lead to overshooting the minimum.

5. **Iteration**: Steps 3 and 4 are repeated for a specified number of iterations or until convergence. Convergence occurs when the gradient is close to zero, indicating that the algorithm has found a minimum or a very flat region of the cost function.

6. **Termination**: The algorithm terminates when it reaches a predefined stopping criterion, such as a maximum number of iterations or when the cost function reaches a specific threshold.

There are variations of Gradient Descent, including:

- **Stochastic Gradient Descent (SGD)**: Updates the parameters using one training example at a time. It can be more computationally efficient but may introduce more noise in the updates.

- **Mini-Batch Gradient Descent**: Updates the parameters using a small random subset (mini-batch) of the training data. It strikes a balance between the efficiency of SGD and stability of Batch Gradient Descent.

- **Batch Gradient Descent**: Updates the parameters using the entire training dataset in each iteration. It can be computationally expensive but provides more stable updates.

Gradient Descent is a fundamental technique for training machine learning models and plays a crucial role in optimizing the parameters to make accurate predictions. Properly tuning the learning rate and selecting the right variant of Gradient Descent is essential for efficient training.
"""
