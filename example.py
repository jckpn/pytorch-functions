import torch
from torchvision import transforms, datasets
from train_model_patience import train_model
from classification_tests import classification_metrics
import networks


# preprocess data transformation - networks usually expect certain input
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(32),
    transforms.ToTensor(),
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root='data/',
        train=True,
        download=True,
        transform=preprocess),
    batch_size=16,
    shuffle=True)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root='data/',
        train=False,
        download=True,
        transform=preprocess),
    batch_size=1,
    shuffle=False)


model = networks.LeNet(num_classes=10)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2',
#                     pretrained=True)

# test before training for comparison
classification_metrics(model, val_loader)


optim = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

device = torch.device("mps") # replace with "cuda" or "cpu" as needed

model = train_model(model, train_loader, optim, loss_fn, val_loader,
                    max_epochs=10, device=device)

classification_metrics(model, val_loader)

# IDEA: nn 1 to check face is facing the right way
#       nn 2 to detect landmark points on front of head
#       nn 3 to predict extra landmark points, using bald heads to train
