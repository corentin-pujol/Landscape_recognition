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
from tqdm import tqdm
import statistics
import matplotlib.pyplot as plt
import pandas as pd

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

def loading_models():
    resnet18 = models.resnet18(pretrained=True)
    mobilenet = models.mobilenet_v2(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    alexnet = models.alexnet(pretrained=True)

    my_models = [resnet18, mobilenet, vgg16, alexnet]
    return my_models

def get_epoch_metrics_mean(train_losses, val_losses, train_accs, val_accs, n_epoch):
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accs = []
    epoch_val_accs = []

    # Calcule la moyenne de perte et d'accuracy pour chaque epoch
    for i in range(n_epoch):
        start_index = i * 250
        end_index = start_index + 250

        epoch_train_loss = sum(train_losses[start_index:end_index]) / 250
        epoch_val_loss = sum(val_losses[start_index:end_index]) / 250
        epoch_train_acc = sum(train_accs[start_index:end_index]) / 250
        epoch_val_acc = sum(val_accs[start_index:end_index]) / 250

        epoch_train_losses.append(epoch_train_loss)
        epoch_val_losses.append(epoch_val_loss)
        epoch_train_accs.append(epoch_train_acc)
        epoch_val_accs.append(epoch_val_acc)

    return epoch_train_losses, epoch_val_losses, epoch_train_accs, epoch_val_accs

def performance_curve(performance_dico, n_epoch):
    fig, axs = plt.subplots(2, len(performance_dico), figsize=(20, 8))
    axs = axs.flatten()

    for i, (k, v) in enumerate(performance_dico.items()):
        train_losses, val_losses, train_accs, val_accs = v["history"]
        #train_losses, val_losses, train_accs, val_accs = get_epoch_metrics_mean(train_losses, val_losses, train_accs, val_accs, n_epoch)
        
        # Plot training and validation losses
        axs[i].plot(train_losses, label='Training Loss')
        axs[i].plot(val_losses, label='Validation Loss')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Loss')
        axs[i].legend()
        axs[i].set_title("Training and validation Loss of " + v["model"].__class__.__name__ + " model")

        # Plot training and validation accuracies
        axs[len(performance_dico) + i].plot(train_accs, label='Training Accuracy')
        axs[len(performance_dico) + i].plot(val_accs, label='Validation Accuracy')
        axs[len(performance_dico) + i].set_xlabel('Epoch')
        axs[len(performance_dico) + i].set_ylabel('Accuracy')
        axs[len(performance_dico) + i].legend()
        axs[len(performance_dico) + i].set_title("Training and validation accuracies of " + v["model"].__class__.__name__ + " model")

    plt.tight_layout()
    plt.show()

    
def evaluate(model, dataset, device, criterion):
    
    model.train(False)
    val_losses = []
    val_accs = []
    #10 000/40 = 250, then to get 250 value for datavalidation we need to have 1500/250 -> 6
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False, num_workers=0)  
    for data in loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)        
        val_losses.append(loss.item())

        # Calcul de l'accuracy
        _, predicted = torch.max(outputs.data, 1)
        acc = (predicted == labels).sum().item() / len(labels)
        val_accs.append(acc)
        
    return val_losses, val_accs


# fonction classique d'entraînement d'un modèle, voir TDs précédents
def train_model(model, loader_train, data_val, optimizer, criterion, device, file_name, scheduler=None, n_epochs=10):
    """
    Params
    --------
        model (PyTorch model): cnn to train
        loader_train (PyTorch dataloader): training dataloader to iterate through
        data_val (PyTorch dataloader): validation dataloader used for early stopping
        optimizer (PyTorch optimizer): optimizer to compute gradients of model parameters
        criterion (PyTorch loss): objective to minimize
        device (torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), allowing to use the GPU if exists
        file_name (str): file path to save the model state dict
        n_epochs (int): maximum number of training epochs

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """
    valid_loss_min = np.Inf
    
    # keep track of training and validation loss each epoch
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    overall_start_time = time.time()
    for epoch in range(n_epochs): # à chaque epochs
        
        loss_train = []
        accuracy_train = []
        
        print("EPOCH % i" % epoch)

        # Set to training
        model.train()
        epoch_start_time = time.time()
        
        for i, data in tqdm(enumerate(loader_train), total=len(loader_train)): # itère sur les minibatchs via le loader apprentissage
            inputs, labels = data
            inputs = inputs.float()  # convert input tensor to float32
            inputs, labels = inputs.to(device), labels.to(device) # on passe les données sur CPU / GPU
            optimizer.zero_grad() # on réinitialise les gradients
            outputs = model(inputs) # on calcule l'output
            
            loss = criterion(outputs, labels) # on calcule la loss
            
            loss.backward() # on effectue la backprop pour calculer les gradients
            optimizer.step() # on update les gradients en fonction des paramètres
            
            loss_train.append(loss.item())

            # Calcul de l'accuracy
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == labels).sum().item() / len(labels)
            accuracy_train.append(acc)
        
        train_losses.append(statistics.mean(loss_train))
        train_accs.append(statistics.mean(accuracy_train))
        # Don't need to keep track of gradients
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()

            model.train(False)
            loss_val, accuracy_val = evaluate(model, data_val, device, criterion)
            val_losses.append(statistics.mean(loss_val))
            val_accs.append(statistics.mean(accuracy_val))
            model.train(True)
            print("{} loss train: {:1.4f}\t acc train: {:1.4f}\t loss val {:1.4f}\t Acc (val): {:.1%}".format(i, statistics.mean(loss_train), statistics.mean(accuracy_train), statistics.mean(loss_val), statistics.mean(accuracy_val)))
            
            if scheduler is not None:
                # Update scheduler
                scheduler.step(statistics.mean(loss_val))

            if statistics.mean(loss_val) < valid_loss_min:
                
                print(file_name + " saved: " + str(statistics.mean(loss_val)) + " < " + str(valid_loss_min))
                valid_loss_min = statistics.mean(loss_val)
                
                # Save model
                torch.save(model, file_name)
                    
        
    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = time.time() - overall_start_time
    
    history = [train_losses, val_losses, train_accs, val_accs]
    return model, history, total_time

def affichage_performance(dico_result):
    
    models = []
    accuracy = []
    inference_times = []
    parameters = []
    for k, v in dico_result.items():
        
        total_params = sum(p.numel() for p in v["model"].parameters())
        models.append(k)
        train_losses, val_losses, train_accs, val_accs = v["history"]
        index_min_loss = val_losses.index(min(val_losses))
        accuracy.append(val_accs[index_min_loss])
        inference_times.append(v['time'])
        parameters.append(total_params)
        
    # Créer un dictionnaire de données
    data = {'Modèles': models,
            'Accuracy': accuracy,
            'Inference time': inference_times,
            'Total number of parameters':parameters}

    # Créer un DataFrame à partir du dictionnaire de données
    df = pd.DataFrame(data)

    # Afficher le DataFrame sous forme de tableau
    print(df) 
    
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














