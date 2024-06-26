import torch
from torchvision import transforms, datasets
from train_model_patience import train_model
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
    batch_size=16,
    shuffle=False)


model = networks.LeNet(num_classes=10)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2',
#                     pretrained=True)


optim = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

device = torch.device("mps") # replace with "cuda" or "cpu" as needed

model = train_model(model, train_loader, optim, loss_fn, val_loader,
                    max_epochs=10, device=device)

# classification_metrics(model, val_loader)
