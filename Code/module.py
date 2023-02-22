# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 15:50:50 2023

@author: coco8
"""

import time
import torch
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
from torchvision import datasets, transforms, models
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

def get_class_labels(root_dir):
    
    class_labels = {}
    for i, directory in enumerate(os.listdir(root_dir)):
        class_labels[i] = directory
    return class_labels

def compute_mean_std(dataset):
    """Compute mean and standard deviation of a dataset"""
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for data in dataset:
        img, _ = data
        mean += img.mean(dim=(1, 2))
        std += img.std(dim=(1, 2))
    mean /= len(dataset)
    std /= len(dataset)
    return mean, std

def dataset(data_transforms, train_path, valid_path, test_path):
    
    # Loading datasets
    train_dataset = ImageFolder(train_path, transform=data_transforms)
    val_dataset = ImageFolder(valid_path, transform=data_transforms)
    test_dataset = ImageFolder(test_path, transform=data_transforms)
    
    return train_dataset, val_dataset, test_dataset


def loader(dataset, batch_size, shuffle = True, num_workers=0):

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def output_layer_adaptation(list_model, num_classes):
    for model in list_model:
        if(model.__class__.__name__=="ResNet"):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    return list_model


def evaluate(model, dataset):
    
    model.train(False)
    
    avg_loss = 0.
    avg_accuracy = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    for data in loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        n_correct = torch.sum(preds == labels)
        
        avg_loss += loss.item()
        avg_accuracy += n_correct
        
    return avg_loss / len(dataset), float(avg_accuracy) / len(dataset)


# fonction classique d'entraînement d'un modèle, voir TDs précédents
def train_model(model, loader_train, data_val, optimizer, criterion, n_epochs=10, PRINT_LOSS = True):
    
    for epoch in range(n_epochs): # à chaque epochs
        print("EPOCH % i" % epoch)
        for i, data in enumerate(loader_train): # itère sur les minibatchs via le loader apprentissage
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # on passe les données sur CPU / GPU
            optimizer.zero_grad() # on réinitialise les gradients
            outputs = model(inputs) # on calcule l'output
            
            loss = criterion(outputs, labels) # on calcule la loss
            if PRINT_LOSS:
                model.train(False)
                loss_val, accuracy = evaluate(my_net, data_val)
                model.train(True)
                print("{} loss train: {:1.4f}\t val {:1.4f}\tAcc (val): {:.1%}".format(i, loss.item(), loss_val, accuracy   ))
            
            loss.backward() # on effectue la backprop pour calculer les gradients
            optimizer.step() # on update les gradients en fonction des paramètres


def transfer_learning(my_net, train_dataset, val_dataset, criterion, optimizer, nb_classes, n_epochs=5):
    
    for param in my_net.parameters():
        param.requires_grad = False
        
    my_net.fc = nn.Linear(in_features=my_net.fc.in_features, out_features=nb_classes, bias=True)
    my_net.to(device) 
    my_net.train(True)
    
    torch.manual_seed(42)
    train_model(my_net, train_dataset, val_dataset, optimizer, criterion, n_epochs=n_epochs)
    

    
def fine_tunning(my_net, train_dataset, val_dataset, criterion, optimizer, nb_classes, list_of_layers_to_finetune, n_epochs=5):
    
    my_net.fc = nn.Linear(in_features=my_net.fc.in_features, out_features=nb_classes, bias=True)
    my_net.to(device)
    
    params_to_update = my_net.parameters()
    list_of_layers_to_finetune=list_of_layers_to_finetune
    params_to_update=[]
    for name,param in my_net.named_parameters():
        if(name in list_of_layers_to_finetune):
            params_to_update.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    my_net_ft.train(True)
    torch.manual_seed(42)
    train_model(my_net, train_dataset, val_dataset, optimizer, criterion, n_epochs=5)

def save_model(my_net, model_path):
    
    torch.save(my_net.state_dict(), model_path)

def load_model(model_class, new_fc, model_path):
        
    model = model_class

    model.fc = new_fc

    model.load_state_dict(torch.load(model_path))
    
    return model


# on définit une fonction pour classifier une image
def classify_image(model, image_path):
    # Open Image
    img = Image.open(image_path)
    
    # Apply the transformations
    img_transformed = data_transforms(img)
    
    # unsqueeze batch dimension, in case you are dealing with a single image
    img_unsquueeze = img_transformed.unsqueeze(0)
    
    # Set model to eval
    model.eval()
    
    # Prediction Time Start
    start_time = time.time()
    
    # Get prediction
    output = model(img_unsquueeze) 
    
    # Prediction Time End
    end_time = time.time()
    
    # Extract prediction 
    score, preds = torch.max(output, 1)
    
    # Compute Classification Time
    classification_time = end_time - start_time
    
    # Compute Classification Score
    confidence = (score / output.abs().sum()) * 100
    confidence = round(confidence.item(),1)

    global class_labels
    label = class_labels[preds.item()]
    
    t = round(classification_time*1000,3)
    
    model_name = model.__class__.__name__
    
    json_result= {"Image" : image_path , "Prediction" : label , "Confidence_prc" : confidence , "Time_ms" : t,
                  "Model" : model_name , "Machine" : torch.cuda.get_device_name(0) }
    
    print( "Prediction: " + str(label) + " - Confidence: " + str(confidence)+"% - Time: " + str(t) + " milliseconds" )
    
    return json_result













