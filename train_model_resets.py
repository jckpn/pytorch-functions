# builds on train_model_patience with weight resets after reaching patience
# warning: big risk of overfitting with this.

# TODO: old code, need to change to match the form of the others (e.g. )


import torch
from torch.utils.data import DataLoader
import math

def create_train_model(
    structure,
    train_set,
    val_set,
    optimizer_fn,
    train_loss_fn,
    val_loss_fn=None,
    batch_size=32,
    initial_lr=0.001,
    weight_decay=0.1,
    max_epochs=20,
    max_resets=10,
    patience_limit=5,
    save_path="unnamed_model.pth",
    load_path=None,
):
    # create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    # batch_size=1 so we can test on individual entries for accuracy

    print("\n RESETS | EPOCHS | TRAIN LOSS | VAL LOSS | BEST VAL LOSS")

    # init training variables
    model = structure
    best_model_val_loss = math.inf
    best_model_num = 1
    best_model_epoch = 1
    model_count = 1
    best_train_history = None

    try:  # allow keyboard/notebook interruption to manually stop training early
        while model_count <= max_resets:
            # reset model params and optimizer for new training run
            model.reset_parameters()
            optimizer = optimizer_fn(
                model.parameters(), lr=initial_lr, weight_decay=weight_decay
            )
            if load_path is not None:
                model.load_state_dict(torch.load(load_path))
            epoch_count = 1
            patience_count = 0
            this_model_history = {"train_loss": [math.inf],
                                  "val_loss": [math.inf],
                                  "best_val_loss": math.inf,
                                  "best_epoch": 1}

            # main training loop
            while epoch_count <= max_epochs:
                model.train()  # set model to training mode
                train_loss = 0
                for x, y in train_loader:
                    optimizer.zero_grad()  # zero out gradients?
                    y_hat = model(x)  # forward pass
                    batch_loss = train_loss_fn(y_hat, y)  # calculate loss
                    batch_loss.backward()  # backward pass
                    optimizer.step()  # optimize/update weights?
                    train_loss += batch_loss.item() / len(train_loader)

                if val_loss_fn is not None:
                    val_loss = val_loss_fn(model, val_loader)
                else:
                    val_loss = default_val_loss(model, val_loader, train_loss_fn)

                this_model_history["train_loss"].append(train_loss)
                this_model_history["val_loss"].append(val_loss)

                if val_loss < this_model_history['best_val_loss'] - 0.0001:
                    # new best epoch for this model
                    this_model_history['best_val_loss'] = val_loss
                    this_model_history['best_epoch'] = epoch_count
                    patience_count = 0

                    if val_loss < best_model_val_loss: # best model so far
                        best_train_history = this_model_history
                        best_model_val_loss = val_loss
                        best_model_num = model_count
                        best_model_epoch = epoch_count
                        torch.save(model.state_dict(), save_path)

                else:  # Early stopping if no improvement to test metrics
                    patience_count += 1
                    if patience_count > patience_limit:
                        break

                print(
                    f"{model_count:7} |",
                    f"{epoch_count:6} |",
                    f"{train_loss:10.3f} |",
                    f"{val_loss:8.3f} |",
                    f"{best_model_val_loss:5.3f} (model {best_model_num}, epoch {best_model_epoch})",
                    end="\r",
                )  # \r ==> next line can overwrite this

                epoch_count += 1

            model_count += 1

    except KeyboardInterrupt:
        pass

    print('\n\nTraining complete (or interrupted).',
          f'Best model (#{best_model_num}, epoch {best_model_epoch}) saved to "{save_path}"')

    # return information about the best model
    best_train_history["train_loss"] = best_train_history["train_loss"][1:]
    best_train_history["total_models"] = model_count
    # remove initial inf values:
    best_train_history["model_num"] = best_model_num
    best_train_history["val_loss"] = best_train_history["val_loss"][1:] 
    
    return best_train_history

def default_val_loss(model, val_loader, loss_fn):
    model.eval()  # set model to evaluation mode
    val_loss = 0
    for x, y in val_loader:
        y_hat = model(x)
        batch_loss = loss_fn(y_hat, y)
        val_loss += batch_loss.item() / len(val_loader)
    return val_loss