import time
import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

print(f" BEGIN RIDGE REGRESSION ".center(96, "="))

print(f"Loading data".ljust(75, "."), end = "", flush = True)

# Load the training (X_train, Y_train) and test (X_test, Y_test) data.
path = "../../data/"
X_train = np.load(f"{path}X_train_transform.npy")
X_test = np.load(f"{path}X_test_transform.npy")
Y_train = pd.read_excel(f"{path}y_train.xlsx", header = None)
Y_test = pd.read_excel(f"{path}test_labels.xlsx", header = None)

# Convert Y_train and Y_test to NumPy arrays.
Y_train = Y_train.to_numpy()[:, 0]
Y_test = Y_test.to_numpy()[:, 0]

# Define the loss function, i.e., squared error.
# This function takes the following inputs:
#   - X [2D NumPy array of features. Size of X is (n, m).]
#   - Y [1D NumPy array of labels. Size of Y is (n, 1).]
#   - beta [1D NumPy array of ridge coefficients. Size of beta is (m, 1).]
# The following value is returned:
#   - Squared L-2 norm of (X @ beta - Y)
def loss_fn(X, Y, beta):
    return cp.pnorm(X @ beta - Y, p=2)**2

# Define the regularizer function, i.e., the squared L-2 norm of the ridge coefficients.
# This function takes the following input:
#   - beta [1D NumPy array of ridge coefficients. Size of beta is (m, 1), where m is the number of rows in X.]
# The following value is returned:
#   - Squared L-2 norm of beta.
def regularizer(beta):
    return cp.pnorm(beta, p=2)**2


def regularizer_l1(beta):
    return cp.norm1(beta)


# Define the objective function, i.e., the ridge estimator function.
# This function takes the following inputs:
#   - X [2D NumPy array of features. Size of X is (n, m).]
#   - Y [1D NumPy array of labels. Size of Y is (n, 1).]
#   - beta [1D NumPy array of ridge coefficients. Size of beta is (m, 1).]
#   - lambd [Penalizer value (float).]
# The following value is returned:
#   - The ridge estimator, i.e., the loss function penalized by the product of the penalizer and regularizer.
def objective_fn(X, Y, beta, lambd, regularizer=regularizer):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)


# Define a function that computes mean squared error.
# This function takes the following inputs:
#   - X [2D NumPy array of features. Size of X is (n, m).]
#   - Y [1D NumPy array of labels. Size of Y is (n, 1).]
#   - beta [1D NumPy array of ridge coefficients. Size of beta is (m, 1).]
# The following value is returned:
#   - Mean squared error of the loss function.
def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value


def add_roc_plot(prediction, labels, name):
    fpr, tpr, thresholds = roc_curve(labels, prediction)
    auc_value = auc(fpr, tpr)
    print(f"{name}, ROC AUC: {round(auc_value, 3)}")
    plt.step(fpr, tpr, where='post', label=f'{name} AUC: {round(auc_value, 3)}')


# Define a function that plots the training and test error for ridge regression as a function of lambda.
# This function takes the following inputs:
#   - train_errors [List of training error values for each candidate lambda.]
#   - test_errors [List of test error values for each candidate lambda.]
#   - lambd_values [1D NumPy array of candidate values for lambda.]
def plot_train_test_errors(train_errors, test_errors, lambd_values, note=''):
    plt.figure(figsize=(10,10))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "century"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.plot(lambd_values, train_errors, color = "indigo", label = "Train Error")
    plt.plot(lambd_values, test_errors, color = "red", label = "Test Error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title(f"MSE for Training and Test Datasets {note}")
    plt.savefig(f'../../plots/plot_train_test_errors{note}.png')


# Define a function that plots the regularization path for ridge regression as a function of lambda.
# This function takes the following inputs:
#   - lambd_values [1D NumPy array of candidate values for lambda.]
#   - beta_values [2D NumPy array of optimal beta values for the candidate lambda.]
def plot_regularization_path(lambd_values, beta_values, note=''):
    num_coeffs = len(beta_values[0])
    plt.figure(figsize=(10,10))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "century"
    plt.rcParams["mathtext.fontset"] = "cm"
    for i in range(num_coeffs):
        plt.plot(lambd_values, [wi[i] for wi in beta_values])
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.ylabel(r"$\beta$ Values")
    plt.xscale("log")
    plt.title(f"Regularization Path {note}")
    plt.savefig(f'../../plots/plot_regularization_path{note}.png')


lambd = cp.Parameter(nonneg=True) # Define the penelizer parameter (i.e., lambda; to be varied).
# Define candidate values for lambda.
lambd_values = np.concatenate([np.zeros(1), np.logspace(-5, 4, 30)])
# Lists to store error values and optimal betas.


data_all = {'l2': dict(), 'l1': dict(), 'l2+l1': dict()}
start = time.time()

# Solve the convex optmization problem (i.e., find the optimal beta values) for each of the candidate lambdas.
cnt = 0
for v in lambd_values:
    print(f"Training ridge classifier".ljust(80, "."), end="", flush=True)
    lambd.value = v
    beta = cp.Variable(X_train.shape[1])
    problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, beta, lambd, regularizer=regularizer)))
    problem.solve(solver=cp.SCS)
    data_all['l2'][lambd.value] = {
        'predictions': X_test @ beta,
        'test_errors': mse(X_test, Y_test, beta),
        'train_errors': mse(X_train, Y_train, beta),
        'betas': beta.value
    }

    beta = cp.Variable(X_train.shape[1])
    problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, beta, lambd, regularizer=regularizer_l1)))
    problem.solve(solver=cp.SCS)
    data_all['l1'][lambd.value] = {
        'predictions': X_test @ beta,
        'test_errors': mse(X_test, Y_test, beta),
        'train_errors': mse(X_train, Y_train, beta),
        'betas': beta.value
    }

    beta = cp.Variable(X_train.shape[1])
    problem = cp.Problem(
        cp.Minimize(objective_fn(
            X_train, Y_train, beta, lambd, regularizer=lambda beta: regularizer(beta) + regularizer_l1(beta)
        ))
    )
    problem.solve(solver=cp.SCS)
    data_all['l2+l1'][lambd.value] = {
        'predictions': X_test @ beta,
        'test_errors': mse(X_test, Y_test, beta),
        'train_errors': mse(X_train, Y_train, beta),
        'betas': beta.value
    }
    cnt += 1
    print(f"Run Execution time = {time.time() - start} s.")
    print(f"Run {cnt} complete.")

end = time.time()

print(f" END RIDGE REGRESSION ".center(96, "="))

print('\n')

print(f"Execution time = {end - start} s.")
sns.set()
sns.set_context('talk')
for regularize_name, regularizer_data in data_all.items():
    for lambda_val, lambda_data in regularizer_data.items():
        plt.figure(figsize=(10,10))
        predictions = lambda_data['predictions'].value
        add_roc_plot(predictions, Y_test, f"{regularize_name} lambda: {lambda_val}")
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.ylim([0.0, 1.01])
        plt.xlim([0.0, 1.01])
        plt.title("Reciever Operating Characteristic (ROC)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'../../plots/roc_{regularize_name}_lambda_{lambda_val}.png')

    lambda_values = regularizer_data.keys()
    train_errors = [regularizer_data[lambda_val]['train_errors'] for lambda_val in lambda_values]
    test_errors = [regularizer_data[lambda_val]['test_errors'] for lambda_val in lambda_values]
    beta_values = [regularizer_data[lambda_val]['betas'] for lambda_val in lambda_values]

    # Plot the training and test error as a function of lambda.
    plot_train_test_errors(train_errors, test_errors, lambd_values, note=regularize_name)
    # Plot the regularization path.
    plot_regularization_path(lambd_values, beta_values, note=regularize_name)
