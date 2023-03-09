import argparse

import torch
from Loader import Loader
from moderArcs.AlexNet import AlexNet

from datetime import datetime
from moderArcs.ResNet import ResNet
from moderArcs.VGG import VGG


def test():
    checkpoint_path = ""
    test_dataset_path = ""
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", required=True)
    parser.add_argument("-checkpoint_path", required=True)
    parser.add_argument("-test_dataset_path", required=True)
    args = parser.parse_args()

    if args.model_name:
        model_name = args.model_name
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    if args.test_dataset_path:
        test_dataset_path = args.test_dataset_path

    # Load the model that we saved at the end of the training loop 
    if model_name in ["VGG11", "VGG13", "VGG16", "VGG19"]:
        model = VGG(model_name)
    elif model_name == "AlexNet":
        model = AlexNet()
    elif model_name == "ResNet50":
        model = ResNet(50)
    else:
        print("Model name \""+model_name+"\" is not valid")
        return
    
    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device\n")
    model.to(device)    # Convert model parameters and buffers to CPU or Cuda  
    model.load_state_dict(torch.load(checkpoint_path))

    # Load the test dataset
    _, __, test_loader = Loader.LoadData(test_dataset_path=test_dataset_path)
     
    running_accuracy = 0 
    total = 0 
    total_inference_time = 0
 
    with torch.no_grad(): 
        for inputs, outputs in test_loader: 
            if torch.cuda.is_available():
                inputs, outputs = inputs.cuda(), outputs.cuda() 
            outputs = outputs.to(torch.float32) 
            starttime = datetime.now()
            predicted_outputs = model(inputs) 
            total_inference_time += (datetime.now() - starttime).total_seconds()
            _, predicted = torch.max(predicted_outputs, 1) 
            total += outputs.size(0) 
            running_accuracy += (predicted == outputs).sum().item() 
 
        print('Accuracy of the model is: %.2f %%' % (100 * running_accuracy / total) + '\nAverage Inference time is: %.4fs' % (total_inference_time/len(test_loader)))   

    
    

if __name__ == "__main__":
    df = test()