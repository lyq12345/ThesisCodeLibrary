import numpy as np

processing_speed = np.array([[0.46, 0.58, 1.09, 0.46, 0.46],
                           [3.86, 4.48, 7.21, 3.86, 3.86],
                           [0.42, 0.55, 1.07, 0.42, 0.42]])

operator_accuracy = np.array([0.45, 0.68, 0.45])

# Define the Boolean matrices A and B, and the transmission time matrix C
X = np.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

Y = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]])

C = np.random.uniform(5, 20, size=(6, 6))

# print(C)

# Calculate matrix D
delay = np.sum(Y.T@np.multiply(Y@X, C), axis=1)
accuracy = np.multiply(np.sum(X, axis=1), operator_accuracy.T)

print(delay)
print(accuracy)
