import torch
import torchvision
import numpy as np
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # For nice progress bar!


torch.manual_seed(28)
torch.cuda.manual_seed(28)
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






#plot the graph
def print_image(dataset,name):

    fig = plt.figure()

    train_number =[]
    for i in range(0,100):
        image, target = dataset[i]

        if target not in train_number:
            train_number.append(target)
            plt.subplot(3,4,len(train_number))
            image = image.numpy()
            plt.tight_layout()
            plt.imshow(image[0], cmap='gray', interpolation='none')
            plt.title("Ground Truth: {}".format(target))
        elif len(train_number) == 10 :
            break
    fig.suptitle('hi')
    fig.show()
    fig.savefig('Original_Image/{}.png'.format(name))


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=25,
            kernel_size=(12, 12),
            stride=(2, 2)
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=25,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2)
        )
        self.fc1 = nn.Linear(1024, num_classes)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x





# Hyperparameters
num_classes = 10
learning_rate = 1e-4
batch_size = 50
num_epochs = 1000

# Loss and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#print parameter
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


#load dataset
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
vis_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)



# print randomly 10 photo in train or test dataset
print_image(train_dataset,'train20')
print_image(test_dataset,'test20')




for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        #print(epoch,loss)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 99:  # print every 200 mini-batches
            print('[epoch%d, mini_batch%5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 100))
            running_loss = 0.0

# Check accuracy on training & test to see how good our model
torch.save(model.state_dict(), 'CNNmodel')



def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            test_scores = model(x)
            _, predictions = test_scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()

    return test_scores


# check accuracy in test data
#
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
#

# visualize layer
model_weights = [] # we will save the conv layer weights in this list
model_children = list(model.children())
fig1 = plt.gcf()
plt.figure(figsize=(20, 17))

for i, filter in enumerate(model_children[0].weight):
    plt.subplot(5, 5, i+1)
    plt.tight_layout()
    plt.imshow(filter[0, :, :].detach().cpu(), cmap='gray')
    plt.title("filter: {}".format(i))


fig1.savefig('Original_Image/vis_filter.png',dpi=100)
plt.show()
plt.draw()




#visualize patches

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# to get the slide window of image by 12x12
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize[1]):
        for x in range(0, image.shape[1], stepSize[0]):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def get_slice(image, stepSize, windowSize):
    slice_sets = []
    for (x, y, window) in sliding_window(image, stepSize, windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
            continue
        slice = image[y:y + windowSize[1], x:x + windowSize[0]]
        slice_sets.append(slice)
    return slice_sets

# graph generating loop.
i = 0
for batch_idx, (data, targets) in enumerate(tqdm(vis_loader)):
    activation = {}
    model.eval()
    # get the
    model.relu1.register_forward_hook(get_activation('relu1'))
    output = model(data.to(device))

    # seting the sliding window size
    data28 = data.squeeze_(0)
    data28 = data28.numpy()
    w = data28.shape[1]
    h = data28.shape[2]
    # kernel size
    (winW, winH) = (12, 12)
    # stride
    stepSize = (2, 2)
    photo_window = get_slice(data28[0], stepSize, (winW, winH))

    act = activation['relu1'].squeeze().cpu().numpy()

    # plot 5 filter
    for v in range(5):
        fig = plt.gcf()

        high_acti = np.argsort((act[v]),axis=None)
        # high_acti = int(torch.argmax(act[v]))

        # plot top 12 activation value
        for d in range(0,12):
            image = photo_window[int(high_acti[-(d+1)])]
            plt.subplot(3, 4, d+1)
            plt.tight_layout()
            plt.imshow(image, cmap='gray', interpolation='none')
            plt.title("activation rank: {}".format(d+1))
            plt.suptitle("test_photo{},filter{}".format(i,v+1), fontsize=14)

        plt.show()

        fig.savefig("Patches_Image/test_photo{},filter{}".format(i,v).format(i), dpi=100)

    i=i+1