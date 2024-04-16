import pickle

from global_name_space import ARGS
from load_data import load_data
from model import LinearRegressionGradientDescent


def save(obj):
    with open(ARGS.path, 'wb') as f:
        pickle.dump(obj, f)


def load():
    with open(ARGS.path, 'rb') as f:
        return pickle.load(f)


def main():
    x_train, y_train, x_test, y_test, _ = load_data()

    a = LinearRegressionGradientDescent(x_train, y_train)
    a.train()
    save(a)

    a = load()
    y_test_pred = a.predict(x_test)
    a.print_fit_measures(y_test, y_test_pred)
    a.plot_y_and_y_pred(y_test, y_test_pred)

if __name__ == "__main__":
    main()