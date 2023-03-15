import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


#Data split 
# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#Setting Device - pytorch magic TODO: learn this
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" 
print(f"using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() #Explain
        self.flatten = nn.Flatten()
        #creating NN structure
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        y = self.flatten(x)
        logits = self.linear_relu_stack(y)
        return logits
    
model = NeuralNetwork().to(device=device)
print(model)

## LEARNING RATE - TODO: como fica a learn_rate com o Adam? 
learning_rate = 1e-3

## LOSS FUNCT
loss_fn = nn.CrossEntropyLoss()

## OPTIMIZER - ADAM 4 the win
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## TRAINING

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Only a print on loss and current batch + acc
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

## TESTING

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    #Inherited function - set model to evaluate mode: idk what that means; something like does not update model
    model.eval()
    test_loss, correct = 0,0

    #Runs no_grad context manager -- stops learning basically, does not run grad_opt
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {100*correct}, avg loss {test_loss}")



epochs = 50

for t in range(epochs):
    print(f"Epoch {t+1}")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done")


torch.save(model.state_dict(), "model_adam.pth")
print("Saving model")

img = None
pred = model(img)