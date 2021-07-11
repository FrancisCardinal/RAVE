import torch 
import os 
from time import localtime, strftime, time
from datetime import timedelta
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.ion()

class Trainer():
    TRAINING_SESSIONS_DIR = 'training_sessions'
    def __init__(self, 
                training_loader, 
                validation_loader, 
                loss_function,
                device,
                model,
                optimizer):
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.loss_function = loss_function
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.training_losses = []
        self.validation_losses = []


    def train_with_validation(self): 
        NB_EPOCHS = 750
        min_validation_loss = np.inf
        start_time = time()
        
        for epoch in range(NB_EPOCHS):
            current_training_loss   = self.compute_training_loss()
            current_validation_loss = self.compute_validation_loss()

            self.training_losses.append(current_training_loss)
            self.validation_losses.append(current_validation_loss)
            self.update_plot()
            
            if min_validation_loss > current_validation_loss:
                print(f'Validation Loss Decreased({min_validation_loss:.6f}--->{current_validation_loss:.6f}) \t Saving The Model')
                min_validation_loss = current_validation_loss
                
                # Saving State Dict
                torch.save(self.model.state_dict(), 'saved_model.pth')
        
        min_training_loss   = min(self.training_losses)
        time_of_completion = strftime("%Y-%m-%d %H:%M:%S", localtime())
        ellapsed_time = str( timedelta( seconds=(time() - start_time) ) )
        figure_title = f'{time_of_completion:s} | ellapsed_time={ellapsed_time:s} | min_validation_loss={min_validation_loss:.6f} | min_training_loss={min_training_loss:.6f}.png'

        plt.savefig(os.path.join(os.getcwd(), Trainer.TRAINING_SESSIONS_DIR, figure_title), dpi = 200)

    
    def compute_training_loss(self):
        self.model.train()

        training_loss = 0.0
        number_of_images = 0 
        for images, labels in tqdm(self.training_loader, 'training', leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Clear the gradients
            self.optimizer.zero_grad()
            # Forward Pass
            predictions = self.model(images)
            # Find the Loss
            loss = self.loss_function(predictions, labels)
            # Calculate gradients 
            loss.backward()
            # Update Weights
            self.optimizer.step()
            # Calculate Loss
            training_loss += loss.item()
            number_of_images += len(images)

        return training_loss/number_of_images
    

    def compute_validation_loss(self):
        with torch.no_grad():
            self.model.eval() 

            validation_loss = 0.0  
            number_of_images = 0 
            for images, labels in tqdm(self.validation_loader, 'validation', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward Pass
                predictions = self.model(images)
                # Find the Loss
                loss = self.loss_function(predictions, labels)
                # Calculate Loss
                validation_loss += loss.item()
                number_of_images += len(images)

            return validation_loss/number_of_images


    def update_plot(self):
        plt.clf()
        plt.plot(range(len(self.training_losses)),   self.training_losses, label='training loss')
        plt.plot(range(len(self.validation_losses)), self.validation_losses, label='validation loss')
        plt.legend(loc="upper left")
        plt.draw()
        plt.gcf().canvas.draw_idle() #Pour éviter que le graphique "vole" le focus et nous empêche de faire autre chose pendant que le réseau s'entraîne
        plt.gcf().canvas.start_event_loop(0.001)