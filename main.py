from back_propagation import BackPropagation
from generate_data import generate_data

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = generate_data()
    print("Shapes of train and test data:")
    print("X_train:", x_train.shape)
    print("Y_train:", y_train.shape)
    print("X_test:", x_test.shape)
    print("Y_test:", y_test.shape)

    bp = BackPropagation(2,10)
    bp.train(x_train, y_train, x_test, y_test)


    

    