# builds on train_model_epochs with patience (i.e. early stopping, training
# halts once validation loss stops decreasing and model with best val loss
# is returned)


import torch
import os
from time import time


def train_model(
    model,
    train_loader,
    optim,
    loss_fn,
    val_loader,
    patience_limit=5,
    max_epochs=10,
    save_path='temp.pt',
    load_path=None,
    device=None):

    print('Training model...')

    epoch_count = 1
    patience_count = 0
    best_val_loss = float('inf')

    if load_path is not None:
        model.load_state_dict(torch.load(load_path))
    
    if device is not None:
        model.to(device)

    try:    # allow keyboard interrupt
        while epoch_count <= max_epochs:
            start_time = time()

            # training loop
            model.train()
            train_loss = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                if device is not None:
                    data = data.to(device)
                    target = target.to(device)

                optim.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optim.step()
                train_loss += loss.item()

                print(f'epoch {epoch_count}',
                      f'batch {batch_idx+1}/{len(train_loader)}',
                      end='\r')

            train_loss /= len(train_loader.dataset)

            # validation loop
            model.eval() # 
            val_loss = 0

            with torch.no_grad():   # disable gradient calcs
                for data, target in val_loader:
                    if device is not None:
                        data = data.to(device)
                        target = target.to(device)

                    output = model(data)
                    val_loss += loss_fn(output, target).item()

            val_loss /= len(val_loader.dataset)

            # check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_count += 1
                if patience_count > patience_limit:
                    break
            
            elapsed_time = time() - start_time

            print(f'epoch {epoch_count}, '
                  f'train_loss = {train_loss:.4f}, '
                  f'val_loss = {val_loss:.4f}, '
                  f'elapsed_time = {elapsed_time:.2f}s')
            
            epoch_count += 1
    
    except KeyboardInterrupt:
        print('KeyboardInterrupt: training stopped early')
        pass

    # load and return best model if saved
    if epoch_count > 0:
        model.load_state_dict(torch.load(save_path))
        print(f'Training complete.')

        if save_path == 'temp.pt':
            # delete temp file
            os.remove(save_path)
            print('WARNING: model was not saved to disk')
        else:
            print(f'Model saved to {save_path}')

    return model