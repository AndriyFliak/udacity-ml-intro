import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image 
import torchvision.transforms as transforms

def load_trained_model(filepath):
    ''' Loading model from file
    '''
    model_dict = torch.load(filepath)
    model = getattr(models, model_dict['network'])()

    layers = []
    prev_layer = model_dict['fc_in']

    for unit in model_dict['fc_hidden_units']:
        layers.append(nn.Linear(prev_layer, unit))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(model_dict['fc_dropout']))
        prev_layer = unit
    layers.append(nn.Linear(prev_layer, model_dict['fc_out']))
    layers.append(nn.LogSoftmax(dim=1))

    classifier = nn.Sequential(*layers)

    if model_dict['network'].startswith('vgg'):
        model.classifier = classifier
    else:
        model.fc = classifier

    state_dict = model_dict['state_dict']
    model.load_state_dict(state_dict)
    model.class_to_idx = model_dict['class_to_idx']
    model.idx_to_class = {v: k for k, v in model_dict['class_to_idx'].items()}
    model.eval()
    return model

def predict(image_path, model, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    im = Image.open(image_path)
    im = process_image(im)
    im = im.unsqueeze(0)
    im = im.to(device)
    with torch.no_grad():
        logps = model(im)
    ps = torch.exp(logps)
    top_ps, top_idx = ps.topk(topk, dim=1)
    top_classes = [model.idx_to_class[cl] for cl in top_idx.flatten().tolist()]
    return top_ps.flatten().tolist(), top_classes

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return im_transforms(image)
