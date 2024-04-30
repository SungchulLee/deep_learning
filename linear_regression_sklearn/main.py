import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from load_data import load_data


def plot_y_and_y_pred(y, y_pred):
    _, ax = plt.subplots(figsize=(12,3))
    ax.plot(y.reshape((-1,)), y.reshape((-1,)), '--r', label="data")
    ax.plot(y.reshape((-1,)), y_pred.reshape((-1,)), 'o', label="pred")
    ax.legend()
    ax.set_xlabel("y")
    plt.show()


def main():
    x_train, y_train, x_test, y_test, _ = load_data(feature_dim=4, error_std=1)

    if 1:
        a = LinearRegression()
    elif 0:
        a = Ridge()
    elif 0:
        a = Lasso()
    elif 1:
        a = ElasticNet()

    a.fit(x_train, y_train)
    #print(a.intercept_)
    #print(a.coef_)

    y_test_pred = a.predict(x_test)
    plot_y_and_y_pred(y_test, y_test_pred)


if __name__ == "__main__":
    main()