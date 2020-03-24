import argparse
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
import torch
import torch.optim as optim
from datetime import datetime

# Parsing arguments
parser = argparse.ArgumentParser(description='Train neural network to detrmine flower type from image')
parser.add_argument('data_directory', help='path to the data')
parser.add_argument('--save_dir', default=False, help='set directory to save checkpoints')
parser.add_argument('--arch', default='resnet152', help='choose architecture: vggX or resnetX')
parser.add_argument('--learning_rate', default=0.003, type=float, help='set learning rate')
parser.add_argument('--hidden_units', default=[1024, 512], type=int, nargs='+', help='set the amount of hidden units, pass multiple numbers for multiple layers')
parser.add_argument('--dropout', default=0.2, type=float, help='set dropout')
parser.add_argument('--epochs', default=20, type=int, help='set epochs')
parser.add_argument('--gpu', action='store_const', const=True, default=False, help='use GPU for training')
args = parser.parse_args()

# Loading and transforming image sets
train_dir = args.data_directory + '/train'
validation_dir = args.data_directory + '/valid'
test_dir = args.data_directory + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset = ImageFolder(train_dir, transform=train_transforms)
validation_dataset = ImageFolder(validation_dir, transform=test_transforms)
test_dataset = ImageFolder(test_dir, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Setting up model
model = getattr(models, args.arch)(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

layers = []
if args.arch.startswith('vgg'):
    fc_in = 25088
else:
    fc_in = 2048
prev_layer = fc_in

for unit in args.hidden_units:
    layers.append(nn.Linear(prev_layer, unit))
    layers.append(nn.RReLU())
    layers.append(nn.Dropout(args.dropout))
    prev_layer = unit
layers.append(nn.Linear(prev_layer, 102))
layers.append(nn.LogSoftmax(dim=1))

classifier = nn.Sequential(*layers)

if args.arch.startswith('vgg'):
    model.classifier = classifier
else:
    model.fc = classifier

device = 'cuda' if args.gpu and torch.cuda.is_available else 'cpu'
model.to(device)

# Training
criterion = nn.NLLLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

epochs = args.epochs
print_every = 10
for e in range(epochs):
    steps = 0
    train_loss = 0
    for images, labels in train_loader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(images)
        loss = criterion(logps, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            
            with torch.no_grad():
                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    loss = criterion(logps, labels)
                    test_loss += loss.item()
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(test_loss/len(validation_loader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loader)), flush=True)
            
            model.train()
            train_loss = 0

# Calculating accuracy on test data
accuracy = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logps = model(images)
        loss = criterion(logps, labels)
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
print("Test Accuracy: {:.3f}".format(accuracy/len(test_loader)), flush=True)

# Saving model
if (args.save_dir):
    model_dict = {'fc_in': fc_in,
                  'fc_hidden_units': args.hidden_units,
                  'fc_out': 102,
                  'fc_dropout': args.dropout,
                  'class_to_idx': train_dataset.class_to_idx,
                  'network': args.arch,
                  'state_dict': model.state_dict()}
    filename = f"{args.save_dir}/{args.arch}-{datetime.timestamp(datetime.now())}.pth"
    torch.save(model_dict, filename)
    print(f"Model saved to {filename}")
