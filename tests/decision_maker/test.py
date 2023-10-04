import numpy as np

processing_speed = np.array([[0.46, 0.58, 1.09, 0.46, 0.46, 0.46],
                           [3.86, 4.48, 7.21, 3.86, 3.86, 3.86],
                           [0.42, 0.55, 1.07, 0.42, 0.42, 3.86]])

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

transmission_rate = np.random.uniform(5, 20, size=(6, 6))

# print(C)

# Calculate matrix D
transmission_delay = np.multiply(Y@X, transmission_rate)
processing_delay = Y @ np.multiply(X, processing_speed)
# delay = transmission_delay + processing_delay
# accuracy = np.multiply(np.sum(X, axis=1), operator_accuracy.T)

# utility = accuracy - np.maximum(0, (delay - 10) / delay)

print(transmission_delay)
print(processing_delay)