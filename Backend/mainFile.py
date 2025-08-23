import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)   #fully connected layer
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)      #for 10 digits, 0-9

    def forward(self, x):
        x = x.view(-1, 28*28)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train_model():
    #mnist 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # mean n std for MNIST
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform) #train data
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform) #test

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #trying 64 batch size rn.
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    for epoch in range(5):  #goign with 5 epchos
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device) 
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    #test and evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"Test Accuracy: {100. * correct / len(test_dataset):.2f}%")

    #saveing model. 
    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model saved to mnist_model.pth")

def predict(image_tensor):
    """
    image_tensor: shape [1, 1, 28, 28] (single grayscale MNIST-like image)
    """
    model = Net().to(device)
    model.load_state_dict(torch.load("mnist_model.pth"))
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        pred = output.argmax(dim=1, keepdim=True)
    return pred.item()

if __name__ == "__main__":
    train_model()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    img, label = test_dataset[0]
    pred = predict(img.unsqueeze(0))
    print(f"True Label: {label}, Predicted: {pred}")