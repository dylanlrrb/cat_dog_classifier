import os.path
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import json
import math
import time

# Load network as feature detector
def build_network(arch="vgg16", out_features=2, hidden_layers=[1000]):
    model = getattr(models, arch)(pretrained=True)

    # Freeze the params from training
    for param in model.parameters():
        param.requires_grad = False

    # Find the number of in features the classifier expects
    try:
        iter(model.classifier)
    except TypeError:
        in_features = model.classifier.in_features
    else:
        in_features = model.classifier[0].in_features
    
    hidden_layers = [in_features] + hidden_layers
    
    # Define how to build a hidden layer
    layer_builder = (
        lambda i, v: (("fc" + str(i)), nn.Linear(hidden_layers[i-1], v)),
        lambda i, v: (("relu" + str(i)), nn.ReLU()),
        lambda i, v: (("drop" + str(i)), nn.Dropout())
    )
    
    # Loop through hidden_layers array and build a layer for each
    layers = [f(i, v) for i, v in enumerate(hidden_layers) if i > 0 for f in layer_builder]
    # Concat the output layer onto the network
    layers += [('fc_final', nn.Linear(hidden_layers[-1], out_features)),
               ('output', nn.LogSoftmax(dim=1))]
    
    classifier = nn.Sequential(OrderedDict(layers))

    # Replace classifier with our classifier
    model.classifier = classifier

    return model

def load_model(path):
    checkpoint = torch.load(path)

    arch = checkpoint['arch']
    out_features = len(checkpoint['class_to_idx'])
    hidden_layers = checkpoint['hidden_layers']

    # For loading, use the same build_network function used to make saved model's network
    model = build_network(arch, out_features, hidden_layers)
    model.load_state_dict(checkpoint['state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open the image
    pil_image = Image.open(image_path)
    
    # Resize the image
    if pil_image.size[1] < pil_image.size[0]:
        pil_image.thumbnail((255, math.pow(255, 2)))
    else:
        pil_image.thumbnail((math.pow(255, 2), 255))
                            
    # Crop the image
    left = (pil_image.width-224)/2
    bottom = (pil_image.height-224)/2
    right = left + 224
    top = bottom + 224
                            
    pil_image = pil_image.crop((left, bottom, right, top))
                            
    # Turn into np_array
    np_image = np.array(pil_image)/255
    
    # Undo mean and std then transpose
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])  
    np_image = (np_image - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if topk < 1:
        topk = 1
        
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    
    model.eval()
    
    # Turn the np_array image into a FloatTensor
    # before running through model
    tensor_img = torch.FloatTensor([process_image(image_path)])
    # Note: model expects an 1D array of tensors of size (3, 244, 244)
    # so that's why I'm putting the result of process_image()
    # in brackets before passing into FloatTensor()
    # in other words, model expects a tensor of size (1, 3, 244, 244)
    # you coud also use `tensor_img = tensor_img.unsqueeze(0)`
    
    tensor_img = tensor_img.to(device)
    
    result = model(tensor_img).topk(topk)
    
    # Take the natural exponent of each probablility
    # to undo the natural log from the NLLLoss criterion
    probs = torch.exp(result[0].data).cpu().numpy()[0]
    # .cpu() to take the tensor off gpu
    # so it can be turned into a np_array
    # .cpu() should not be called if predicting on a cpu already
    idxs = result[1].data.cpu().numpy()[0]
    
    return(probs, idxs)

loaded_model = load_model('./model/checkpoint.pt')

def classify(image_path):
  start_time = time.time()
  # Get the probabilties and indices from passing the image through the model
  probs, idxs = predict(image_path, loaded_model, topk=2)
  end_time = time.time()

  return {
    "mapping":{"0": "Cat 🐱", "1": "Dog 🐶"},
    "ranking":list(map(lambda x: str(x) ,idxs)),
    "certainty":list(map(lambda x: str(round(x * 100, 1)) ,probs)),
    "speed":f"{round((end_time-start_time) * 1000)}"
  }