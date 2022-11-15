import copy
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import ImageClassifierDataset
from load_images import load_images
from network import SimpleNet


def adjust_learning_rate(epoch):
    lr = 0.001
    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_models(epoch):
    torch.save(model.state_dict(), "cnn{}.model".format(epoch))
    print("Improved model saved")


def getNumberOfPhotos(training):
    if training:
        return len(os.listdir('./Maps/ColdFront')) + len(os.listdir('./Maps/WarmFront')) + \
               len(os.listdir('./Maps/MixedFront')) + len(os.listdir('./Maps/NoFront'))
    else:
        return len(os.listdir('./Test/NotTrained/ColdFront')) + len(os.listdir('./Test/NotTrained/WarmFront')) + \
               len(os.listdir('./Test/NotTrained/MixedFront')) + len(os.listdir('./Test/NotTrained/NoFront'))


accuracies_by_classes = []
classes_names = ['none', 'cold', 'warm', 'mixed']
confmat = confusion_matrix([], [], labels=classes_names)
best_confmat = confusion_matrix([], [], labels=classes_names)


def test():
    global confmat
    model.eval()
    test_acc = 0.0
    labels_for_matrix = []
    predictions = []
    for i, (images, labels) in enumerate(test_loader):
        # Predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        # prediction = prediction.cpu().numpy()
        labels_for_matrix.extend(labels.data)
        predictions.extend(prediction)
        test_acc += torch.sum(torch.eq(prediction, labels.data))

    confmat = confusion_matrix(labels_for_matrix, predictions)
    accuracies_by_classes.append(confmat.diagonal() / confmat.sum(axis=1))
    # Compute the average acc and loss over all test images
    test_acc = test_acc / getNumberOfPhotos(False)
    return test_acc


test_accuracies = []
losses = []


def train(num_epochs):
    global best_confmat
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("training...")
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the train set
            outputs = model(images)
            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().data.item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data)

        # Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        # Compute the average acc and loss over all 50000 training images
        number = getNumberOfPhotos(True)
        train_acc = train_acc / number
        train_loss = train_loss / number
        losses.append(train_loss)

        # Evaluate on the test set
        test_acc = test()
        test_accuracies.append(test_acc)

        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc
            best_confmat = copy.deepcopy(confmat)

        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,
                                                                                        test_acc))


cold_front_train = "./Maps/ColdFront"
warm_front_train = "./Maps/WarmFront"
mixed_front_train = "./Maps/MixedFront"
no_fronts_train = "./Maps/NoFront"
cold_front_test = "./Test/NotTrained/ColdFront"
warm_front_test = "./Test/NotTrained/WarmFront"
mixed_front_test = "./Test/NotTrained/MixedFront"
no_fronts_test = "./Test/NotTrained/NoFront"

batch_size = 32

images, classes = load_images(cold_front_train, warm_front_train, mixed_front_train, no_fronts_train)
train_set = ImageClassifierDataset(images, classes)
# Create a loader for the training set
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

images, classes = load_images(cold_front_test, warm_front_test, mixed_front_test, no_fronts_test)
test_set = ImageClassifierDataset(images, classes)
# Create a loader for the test set, note that both shuffle is set to false for the test loader
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

# Create model, optimizer and loss function
model = SimpleNet(num_classes=4)

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()

if __name__ == "__main__":
    train(50)
    plt.plot(test_accuracies, color='blue')
    plt.show()
    # print_confusion_matrix(confmat, ['none', 'cold', 'warm', 'mixed'])
    # df_cm = pd.DataFrame(, index=[i for i in classes_names],
    #                      columns=[i for i in classes_names])

    plt.plot([acc[0] for acc in accuracies_by_classes], color='orange', label='none')  # none
    plt.plot([acc[1] for acc in accuracies_by_classes], color='blue', label='cold')  # cold
    plt.plot([acc[2] for acc in accuracies_by_classes], color='red', label='warm')  # warm
    plt.plot([acc[3] for acc in accuracies_by_classes], color='green', label='mixed')  # mixed
    plt.legend(loc="upper left")
    plt.show()

    plt.figure(figsize=(12, 7))
    sns.heatmap(best_confmat/np.sum(best_confmat) * 4, annot=True, fmt='.2%', yticklabels=classes_names,
                xticklabels=classes_names)
    plt.savefig('output.png')
