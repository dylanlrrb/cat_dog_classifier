import torch
import numpy as np
from torch import nn
import torchvision
from collections import OrderedDict
from PIL import Image
import numpy as np
import json
import math
import time

# Load network as feature detector
def build_network(architecture, out_features, hidden_layers, label_mapping, log_model=False):

  model = getattr(torchvision.models, architecture)(pretrained=True)
  for param in model.parameters():
    param.requires_grad = False

  # Find the number of in features the classifier expects
  try:
      iter(model.classifier)
  except TypeError:
      in_features = model.classifier.in_features
  else:
      in_features = model.classifier[0].in_features

  hidden = [in_features] + hidden_layers

  layers = []
  for i, (x, y) in enumerate(zip(hidden[:-1], hidden[1:])):
    layers.append((f'fc{i}', nn.Linear(x, y)))
    layers.append((f'relu{i}', nn.ReLU()))
    layers.append((f'dropout{i}', nn.Dropout(p=0.2)))
  layers.append(('fc_final', nn.Linear(hidden_layers[-1], out_features)))
  layers.append(('log_output', nn.LogSoftmax(dim=1)))

  classifier = nn.Sequential(OrderedDict(layers))
  if log_model:
    print('Classifier:', classifier)

  classifier.out_features = out_features
  classifier.hidden_layers = hidden_layers
  classifier.label_mapping = label_mapping
  model.architecture = architecture
  
  model.classifier = classifier
  return model

def load(filepath):
  checkpoint = torch.load(filepath)
  model = build_network(checkpoint['architecture'],
                        checkpoint['out_features'],
                        checkpoint['hidden_layers'],
                        checkpoint['label_mapping'])
  model.load_state_dict(checkpoint['state_dict'])

  return model

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


loaded_model = load('./model/checkpoint.pth')
device = torch.device("cpu")
loaded_model.to(device)
label_mapping = loaded_model.classifier.label_mapping


def classify(image_path):
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
    np_image = np.transpose(np_image, (2, 0, 1))

    with torch.no_grad():
        loaded_model.eval()
        # Turn the np_array image into a FloatTensor before running through model
        img_tensor = torch.FloatTensor([np_image]).to(device)

        start_time = time.time()
        top_p, top_class = torch.exp(loaded_model.forward(img_tensor)).topk(len(label_mapping), dim=1)
        end_time = time.time()
        return {
            "mapping":label_mapping,
            "ranking":list(map(lambda x: str(x), top_class.squeeze().numpy())),
            "certainty":list(map(lambda x: str(round(x * 100, 1)), top_p.squeeze().numpy())),
            "speed":f"{round((end_time-start_time) * 1000)}"
        }
