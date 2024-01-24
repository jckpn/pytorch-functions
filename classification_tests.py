# common tests for classification networks: accuracy, confusion matrix, etc.

import torch
import matplotlib.pyplot as plt

def classification_metrics(model, test_loader, print_results=True):
    model.eval()

    tests = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for x, y in test_loader:
        y_hat = model(x)
        y_hat = torch.argmax(y_hat, dim=1)

        if y_hat == 1 and y == 1:
            tp += 1
        elif y_hat == 0 and y == 0:
            tn += 1
        elif y_hat == 1 and y == 0:
            fp += 1
        elif y_hat == 0 and y == 1:
            fn += 1
        
        tests += 1
    
    p_acc = tp / (tp + fn)
    n_acc = tn / (tn + fp)
    b_acc = (p_acc + n_acc) / 2

    if print_results:
        print(f' --> Positive accuracy: {p_acc*100:.2f}% ({tp}/{tp+fn})')
        print(f' --> Negative accuracy: {n_acc*100:.2f}% ({tn}/{tn+fp})')
        print(f' --> Balanced accuracy: {b_acc*100:.2f}%')
    
    # plot confusion matrix
    fig, ax = plt.subplots()
    ax.imshow([[tp, fp], [fn, tn]], cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Positive', 'Negative'])
    ax.set_yticklabels(['Positive', 'Negative'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion matrix')
    # set size
    fig.set_size_inches(2, 2)
    plt.show()


def balanced_acc_loss(model, val_loader):
    model.eval()

    tests = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for x, y in val_loader:
        y_hat = model(x)
        y_hat = torch.argmax(y_hat, dim=1)

        if y_hat == 1 and y == 1:
            tp += 1
        elif y_hat == 0 and y == 0:
            tn += 1
        elif y_hat == 1 and y == 0:
            fp += 1
        elif y_hat == 0 and y == 1:
            fn += 1
        
        tests += 1
    
    p_acc = tp / (tp + fn)
    n_acc = tn / (tn + fp)
    b_acc = (p_acc + n_acc) / 2

    return 1-b_acc