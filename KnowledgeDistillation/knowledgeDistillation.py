import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Check if GPU is available, and if not, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Below we are preprocessing data for CIFAR-10. We use an arbitrary batch size of 128.
transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading the CIFAR-10 dataset:
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

#Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)


class DeepNN(nn.Module):
    def __init__(self,num_classes=10):
        super(DeepNN, self).__init__()
        self.features = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=128,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512,num_classes)
        )
    
    def forward(self, x):

        layer1 = self.features(x)
        layer2 = torch.flatten(layer1,1)
        layer3 = self.classifier(layer2)
        return layer3
    

class LightNN(nn.Module):
    def __init__(self,num_classes=10):
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256,num_classes)
        )
    
    def forward(self, x):

        layer1 = self.features(x)
        layer2 = torch.flatten(layer1,1)
        layer3 = self.classifier(layer2)
        return layer3
    

def train(model, train_loader, epochs, learning_rate, device):
    
    criterion  = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        for input, label in train_loader:

            input, label = input.to(device), label.to(device)

            """In PyTorch, for every mini-batch during the training phase, we typically want to explicitly set the gradients to zero before starting to do backpropagation (i.e., updating the Weights and biases) because PyTorch accumulates the gradients on subsequent backward passes. This accumulating behavior is convenient while training RNNs or when we want to compute the gradient of the loss summed over multiple mini-batches. So, the default action has been set to accumulate (i.e. sum) the gradients on every loss.backward() call.

                Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly. Otherwise, the gradient would be a combination of the old gradient, which you have already used to update your model parameters and the newly-computed gradient. It would therefore point in some other direction than the intended direction towards the minimum (or maximum, in case of maximization objectives)."""
            
            optimizer.zero_grad()
            
            output = model(input)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")





torch.manual_seed(42)
nn_deep = DeepNN(num_classes=10).to(device)
train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

test_accuracy_deep = test(nn_deep, test_loader, device)

# Instantiate the lightweight network:
nn_light = LightNN(num_classes=10).to(device)

train(nn_light, train_loader, epochs=10, learning_rate=0.001, device=device)

def knowledgedistillation(teacher, student, trainloader, soft_weighted_loss , weighted_ce_loss, learning_rate,epochs, T, ):

  criteria = nn.CrossEntropyLoss()
  optimizer = optim.Adam(params=student.parameters(),lr=learning_rate)

  teacher.eval()
  student.train()

  for epoch in range(epochs):

    running_loss = 0 
    for input, label in trainloader:

      input, label = input.to(device), label.to(device)

      optimizer.zero_grad()

      with torch.no_grad():

        teacher_logits = teacher(input)

      student_logits = teacher(input)

      #Losses

      soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
      soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

      # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
      soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

      ce_loss = criteria(student_logits,label)

      loss = soft_weighted_loss*soft_targets_loss + weighted_ce_loss*ce_loss

      loss.backward()
      optimizer.step()

      running_loss+= loss.detach()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

knowledgedistillation(nn_deep, nn_light, train_loader, soft_weighted_loss=0.25,weighted_ce_loss=0.75,learning_rate=0.001, T=2,epochs=10 )

      

