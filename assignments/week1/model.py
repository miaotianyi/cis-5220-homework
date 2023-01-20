import numpy as np


class LinearRegression:
    """
    The base linear regression model that uses closed-form formula for fitting.
    """

    w: np.ndarray
    b: float

    def __init__(self) -> None:
        # lazy initialization: set shape after seeing first (X, y)
        self.b = 0.0
        self.w = np.array([])

    @staticmethod
    def _add_ones_column(X: np.ndarray) -> np.ndarray:
        return np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit a linear regression model using closed-form formula.

        Parameters
        ----------
        X : np.ndarray of shape [n_samples, n_features]
            The input design matrix for feature values

        y : np.ndarray of shape [n_samples]
            The array of target values to predict

        """
        # from sklearn.linear_model import LinearRegression as LR
        # model = LR()
        # model.fit(X, y)
        # self.b, self.w = model.intercept_, model.coef_
        # return
        X = self._add_ones_column(X)    # add column of 1s for bias
        w = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        self.b, self.w = w[0], w[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(self, X: np.ndarray, y: np.ndarray,
            lr: float = 0.01, epochs: int = 1000) -> None:
        """
        Fit a linear regression model using gradient descent.

        Parameters
        ----------
        X : np.ndarray of shape [n_samples, n_features]
            The input design matrix for feature values

        y : np.ndarray of shape [n_samples]
            The array of target values to predict

        lr : float, default: 0.01
            Learning rate for gradient descent.

        epochs : int, default: 1000
            The number of epochs (number of passes through the dataset)

        """
        n_samples, n_features = X.shape
        w = np.zeros(n_features + 1)
        w[1:] = np.random.randn(n_features)

        X = self._add_ones_column(X)

        # objective function: J(w) = 1/n * 1/2 * (y_hat - y)**2
        # y_hat = X @ w
        # 1/2 is there to simplify computation
        # dJ/dy_hat = 1/n * 1/2 * 2 * (y_hat - y) = 1/n * (y_hat - y)
        # dy_hat/dw = X

        # losses = []

        for i in range(epochs):
            y_hat = X @ w
            # losses.append(np.mean((y - y_hat)**2))
            # grad = ((y_hat - y) @ X) / n
            grad = (y_hat - y) @ X
            w -= (lr / n_samples) * grad

        # from matplotlib import pyplot as plt
        # plt.plot(np.log(losses))
        # plt.show()
        self.b, self.w = w[0], w[1:]

    # def predict(self, X) -> np.ndarray:
    #     return super(GradientDescentLinearRegression, self).predict(X)


# m1 = LinearRegression()
# m2 = GradientDescentLinearRegression()
# x = np.random.rand(100, 5)
# y = x @ np.arange(5)
# m1.fit(x, y)
# m2.fit(x, y, lr=0.2, epochs=10**3)
# print(m1.w, m1.b)
# print(m2.w, m2.b)

