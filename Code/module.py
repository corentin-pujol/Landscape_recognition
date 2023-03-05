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

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


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

def dataset_with_data_augmentation(train_data_transforms, data_transforms, train_path, valid_path, test_path):
    
    # Loading datasets
    train_dataset = ImageFolder(train_path, transform=train_data_transforms)
    val_dataset = ImageFolder(valid_path, transform=data_transforms)
    test_dataset = ImageFolder(test_path, transform=data_transforms)
    
    return train_dataset, val_dataset, test_dataset


def loader(dataset, batch_size, shuffle = True, num_workers=0):

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# def output_layer_adaptation(list_model, num_classes):
#     for model in list_model:
#         if(model.__class__.__name__=="ResNet"):
#             num_ftrs = model.fc.in_features
#             model.fc = nn.Linear(num_ftrs, num_classes)
#         else:
#             num_ftrs = model.classifier[-1].in_features
#             model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
#     return list_model

def output_layer_adaptation(list_model, num_classes, dropout_prob=0.3, l2_reg=0.01):
    """
    
    dropout_prob est le taux de dropout, c'est-à-dire la probabilité qu'un neurone soit mis à zéro pendant l'entraînement. Cela permet d'éviter l'overfitting en forçant le réseau à ne pas trop dépendre de certains neurones spécifiques.

l2_reg est le paramètre de régularisation L2, qui permet de contrôler la complexité du modèle en ajoutant une pénalité sur les grands poids. Cela peut également aider à éviter l'overfitting.
    """
    
    for model in list_model:
        if model.__class__.__name__ == "ResNet":
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(256, num_classes)
            )
            # Add L2 regularization
            for param in model.fc.parameters():
                param.requires_grad = True
                if len(param.shape) == 2:
                    param.register_hook(lambda grad, l2_reg=l2_reg: grad + l2_reg * torch.mean(param))
        else:
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(256, num_classes)
            )
            # Add L2 regularization
            for param in model.classifier.parameters():
                param.requires_grad = True
                if len(param.shape) == 2:
                    param.register_hook(lambda grad, l2_reg=l2_reg: grad + l2_reg * torch.mean(param))
                
    return list_model


def loading_models():
    resnet18 = models.resnet18(pretrained=True)
    mobilenet = models.mobilenet_v2(pretrained=True)
    #vgg16 = models.vgg16(pretrained=True)
    #alexnet = models.alexnet(pretrained=True)

    my_models = [resnet18, mobilenet]
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
    
    
def loading_models_to_finetune():

    my_models = []
    list_parameters = ["MobileNetV2_SGD_scheduler.pt", "ResNet_SGD_scheduler.pt"]
    
    for i in range(len(list_parameters)):
        
        model = torch.load(list_parameters[i])
        
        for param in model.parameters():
            param.requires_grad = False

        # Geler les paramètres de la dernière couche
        if model.__class__.__name__ == "ResNet":
            for name, module in model.named_children():
                if name in ['layer4']:
                    for param in module.parameters():
                        param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                if "features.18" in name or "features.19" in name:
                    param.requires_grad = True
                
        my_models.append(model)
        
    return my_models

    
def global_training(dico_transferlearning, dico_finetunning):
    
    fig, axs = plt.subplots(2, len(dico_transferlearning), figsize=(20, 8))
    axs = axs.flatten()

    for i, (k, v) in enumerate(dico_transferlearning.items()):
        train_losses_1, val_losses_1, train_accs_1, val_accs_1 = v["history"]
        train_losses_2, val_losses_2, train_accs_2, val_accs_2 = dico_finetunning[k]["history"]
        
        train_losses = train_losses_1 + train_losses_2
        val_losses = val_losses_1 + val_losses_2
        train_accs = train_accs_1 + train_accs_2
        val_accs = val_accs_1 + val_accs_2
        
        axs[i].axvline(x=20, color='r', label='fine-tuning start') #Red line to separate the two steps
        
        # Plot training and validation losses
        axs[i].plot(train_losses, label='Training Loss')
        axs[i].plot(val_losses, label='Validation Loss')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Loss')
        axs[i].legend()
        axs[i].set_title("Training and validation Loss of " + v["model"].__class__.__name__ + " model")
        
        axs[len(dico_transferlearning) + i].axvline(x=20, color='r', label='fine-tuning start') #Red line to separate the two steps

        # Plot training and validation accuracies
        axs[len(dico_transferlearning) + i].plot(train_accs, label='Training Accuracy')
        axs[len(dico_transferlearning) + i].plot(val_accs, label='Validation Accuracy')
        axs[len(dico_transferlearning) + i].set_xlabel('Epoch')
        axs[len(dico_transferlearning) + i].set_ylabel('Accuracy')
        axs[len(dico_transferlearning) + i].legend()
        axs[len(dico_transferlearning) + i].set_title("Training and validation accuracies of " + v["model"].__class__.__name__ + " model")

    plt.tight_layout()
    plt.show()


def test_models(model, test_data):
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Évaluer le modèle sur l'ensemble de test
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true += labels.tolist()
            y_pred += predicted.tolist()
            
    
    # Calculer la matrice de confusion et les métriques de performance
    confusion = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    
    # Créer un dictionnaire contenant les métriques de performance
    metrics = {'Model':[model.__class__.__name__], 'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall], 'F1 score': [f1]}
    # Convertir le dictionnaire en DataFrame
    df_metrics = pd.DataFrame(metrics)
   
    return df_metrics, confusion


def loading_models_to_test():
    my_models = []
    list_parameters = ["MobileNetV2_fine_tunning.pt", "ResNet_fine_tunning.pt"]
    
    for i in range(len(list_parameters)):
        
        model = torch.load(list_parameters[i])
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.eval()
        my_models.append(model)
    return my_models

    
