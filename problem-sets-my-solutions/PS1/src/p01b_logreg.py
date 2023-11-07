import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        for i in range(self.max_iter):
            _theta = self.theta.copy()
            x_theta = x.dot(self.theta)
            hessian = np.linalg.multi_dot([x.T, np.diag(util.logit_(x_theta)), x])
            grad = x.T.dot(y - util.logit(x_theta))
            self.theta = self.theta + np.linalg.inv(hessian).dot(grad)

            error = np.sqrt(np.sum((_theta - self.theta) * (_theta - self.theta)))
            if error < self.eps:
                break

# *** START CODE HERE ***
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        # *** END CODE HERE ***
