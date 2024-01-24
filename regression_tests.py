# tests for regression models (MAE, MAE %, plot predictions vs actual, etc.)
# TODO: option to use specific device


import matplotlib.pyplot as plt


def regression_mae(model, test_loader, denorm_fn=lambda y: y, print_results=True):
    model.eval()

    total = 0
    error = 0
    error_percent = 0

    for x, y in test_loader:
        y = y.item()
        y_hat = model(x).item()
        y_hat = denorm_fn(y_hat)

        error += abs(y_hat - y)
        error_percent += abs((y_hat - y) / y) if y != 0 else 0
        total += 1 if y != 0 else 0 # don't count if y is 0 since inf

    mae = error / total
    mae_percent = error_percent / total
    
    if print_results:
        print(f'Model MAE: {mae:.2f}')
        print(f'Model MAE %: {mae_percent*100:.2f}%')
        print(f'Examples:')

        all_y = []
        all_y_hat = []

        for x, y in test_loader:
            y = y.item()
            y_hat = model(x).item()
            y_hat = denorm_fn(y_hat)
            all_y.append(y)
            all_y_hat.append(y_hat)
        
        # plot predictions vs actual
        fig, ax = plt.subplots()
        ax.scatter(all_y, all_y_hat)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Predictions vs actual')
        plt.show()