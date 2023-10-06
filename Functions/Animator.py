# animator.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def update_plot(i, X, y, model, batch_size):
    # Calculate the predicted values for the first batch_size data points in the current batch
    y_pred = model.pred_probs(X[i:i + batch_size])

    # Add a dummy value to the end of the predicted values array
    y_dummy = np.zeros(100 - len(y_pred))
    y_pred = np.append(y_pred, y_dummy)

    # Reshape the predicted values to have the same shape as the X array
    y_pred = y_pred.reshape((100, 1))

    # Resize the X array to have the same shape as the y_pred array
    X = X[:, None]

    # Convert the y_pred sequence to a single NumPy array
    y_pred = np.asarray(y_pred)
    y_pred = y_pred.astype(np.float32)

    # Plot the predicted values
    y_pred = np.concatenate(y_pred)
    plt.plot((X[:i + batch_size], y_pred), color="red")

    # Add a dummy value to the y array to make sure that the first dimensions are the same
    y_dummy = np.zeros(batch_size)
    y = np.concatenate((y, y_dummy))

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Model Training Process")















