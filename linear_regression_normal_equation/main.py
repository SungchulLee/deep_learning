from load_data import load_data
from model import LinearRegressionNormalEquation


def main():
    x_train, y_train, x_test, y_test, _ = load_data(feature_dim=4,error_std=0.2)
    # print(coeff)
    # print(x_train.shape) # (100,4)
    # print(y_train.shape) # (100,1) or (100,)
    # print(x_test.shape)  # (100,4)
    # print(y_test.shape)  # (100,1) or (100,)

    a = LinearRegressionNormalEquation(x_train, y_train)
    a.train()
    #print(a.coeff)

    y_test_pred = a.predict(x_test)
    a.print_fit_measures(y_test, y_test_pred)
    a.plot_y_and_y_pred(y_test, y_test_pred)

if __name__ == "__main__":
    main()