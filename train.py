import argparse
import os
from torch import nn
import torch
from Loader import Loader
from moderArcs.AlexNet import AlexNet
from moderArcs.ResNet import ResNet
from moderArcs.VGG import VGG
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def saveModel(model_name, epoch_no, timeStamp, model):
    if(os.path.isdir("./"+model_name)):
        if not os.path.isdir("./"+model_name+"/"+ str(timeStamp)):
            os.mkdir("./"+model_name+"/"+ str(timeStamp)) 
    else:
        os.mkdir("./"+model_name) 
        os.mkdir("./"+model_name+"/"+ str(timeStamp)) 
    
    path = "./"+model_name+"/"+ str(timeStamp)+"/saved_model_"+str(epoch_no)+".pth"
    torch.save(model.state_dict(), path)

def train():
    model_name = "VGG11"
    train_dataset_path = ""
    validation_dataset_path = ""
    learning_rate = 0.001
    number_of_epoch = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", required=True)
    parser.add_argument("-train_dataset_path", required=True)
    parser.add_argument("-validation_dataset_path", required=True)
    parser.add_argument("-learning_rate", required=True)
    parser.add_argument("-number_of_epoch", required=True)
    parser.add_argument("-optimizer", required=True)
    parser.add_argument("-draw", required=True)
    args = parser.parse_args()

    if args.model_name:
        model_name = args.model_name
    if args.train_dataset_path:
        train_dataset_path = args.train_dataset_path
    if args.validation_dataset_path:
        validation_dataset_path = args.validation_dataset_path
    if args.learning_rate:
        learning_rate = float(args.learning_rate)
    if args.number_of_epoch:
        number_of_epoch = int(args.number_of_epoch)

    if model_name in ["VGG11", "VGG13", "VGG16", "VGG19"]:
        model = VGG(model_name)
    elif model_name == "AlexNet":
        model = AlexNet()
    elif model_name == "ResNet50":
        model = ResNet(50)
    else:
        print("Model name \""+model_name+"\" is not valid")
        return
    
    timeStamp = datetime.datetime.now().timestamp()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device\n")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        print("Optimizer name is not valid")
        return

    train_loader, validate_loader, _ = Loader.LoadData(
        train_dataset_path, validation_dataset_path, batch_size=32)
    
    best_accuracy = 0.0

    print("Begin training...")
    for epoch in range(1, number_of_epoch+1):
        epochs = []
        train_losses = []
        val_losses = []
        accuracies = []
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_vall_loss = 0.0
        total = 0

        # Training Loop
        for inputs, outputs in train_loader:
            if torch.cuda.is_available():
                inputs, outputs = inputs.cuda(), outputs.cuda()
            optimizer.zero_grad() 
            predicted_outputs = model(inputs)
            train_loss = loss_fn(predicted_outputs, outputs)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()

        # Calculate training loss value
        train_loss_value = running_train_loss/len(train_loader)

        # Validation Loop
        with torch.no_grad():
            model.eval()
            for inputs, outputs in validate_loader:
                if torch.cuda.is_available():
                    inputs, outputs = inputs.cuda(), outputs.cuda()
                predicted_outputs = model(inputs)
                val_loss = loss_fn(predicted_outputs, outputs)

                # The label with the highest value will be our prediction
                _, predicted = torch.max(predicted_outputs, 1)
                running_vall_loss += val_loss.item()
                total += outputs.size(0)
                running_accuracy += (predicted == outputs).sum().item()

        # Calculate validation loss value
        val_loss_value = running_vall_loss/len(validate_loader)

        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.
        accuracy = (100 * running_accuracy / total)

        # Save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel(model_name, epoch, timeStamp, model)
            best_accuracy = accuracy

        # Print the statistics of the epoch
        print('#Epoch ', epoch, '--->Training Loss is: %.4f' % train_loss_value,
              '\tValidation Loss is: %.4f' % val_loss_value, '\tAccuracy is %d %%' % (accuracy))
        epochs.append(epoch)
        train_losses.append(train_loss_value)
        val_losses.append(val_loss_value)
        accuracies.append(accuracy)

        # Save the statistics of the training
        if epoch == 1:
            header = True
        else:
            header = False
        df = pd.DataFrame({'Epoch': epochs, 'Train Loss': train_losses, 'Validation Loss': val_losses, 'Accuracy': accuracies})
        df.to_csv("./"+model_name+"/"+ str(timeStamp)+"/statistics.csv", mode='a', index=False, header=header)

    if args.draw:
        drawStatistics("./"+model_name+"/"+ str(timeStamp)+"/statistics.csv")

def drawStatistics(csv_path):
    df = pd.read_csv(csv_path)
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(df['Epoch'], df['Accuracy'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    

if __name__ == "__main__":
    df = train()

