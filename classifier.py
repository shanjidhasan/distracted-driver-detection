import torch
from torch import nn
import torchvision.transforms.functional as fn
from torchvision import models
from datetime import datetime
import torch
import torchvision.transforms as T
from moderArcs.AlexNet import AlexNet
from moderArcs.VGG import VGG



def classify(image, model_name):
    
    # device = torch.device('cpu')
    if model_name == 'alexnet':
        model = AlexNet()
        # model.to(device) 
        model.load_state_dict(torch.load('./models/alexnet.pth', map_location=torch.device('cpu')))
        model.eval()
    elif model_name == 'vgg16':
        model = VGG("VGG16")
        # model.to(device) 
        model.load_state_dict(torch.load('./models/vgg16.pth', map_location=torch.device('cpu')))
        model.eval()
    elif model_name == 'vgg19':
        model = VGG("VGG19")
        # model.to(device) 
        model.load_state_dict(torch.load('./models/vgg19.pth', map_location=torch.device('cpu')))
        model.eval()
    elif model_name == 'resnet50':
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        model.load_state_dict(torch.load("./models/model-driver", map_location=torch.device('cpu')))
        model.eval()
        # model.cuda()

    

    # load the image in rgb format with size 224x224 using PIL
    img = T.Compose([T.Resize((224,224)), T.ToTensor()])(image.convert('RGB'))

    mean = [0.3141, 0.3803, 0.3729]
    std = [0.2814, 0.3255, 0.3267]
    img = fn.normalize(img, mean, std)

    # add a batch dimension
    img = img.unsqueeze(0)
    # predict the class of the image
    # calculate inference time
    # img = img.to(device)
    
    starttime = datetime.now()
    pred = model(img)
    inference_time = (datetime.now() - starttime).total_seconds()
    # # get the index of the max log-probability
    pred = pred.argmax(dim=1, keepdim=True)
    del model
    return pred.item(), inference_time



