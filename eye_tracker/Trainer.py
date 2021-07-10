import torch 
from livelossplot import PlotLosses

class Trainer():
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
        self.liveloss = PlotLosses()

    def train_with_validation(self): 
        NB_EPOCHS = 50
        min_validation_loss = np.inf
        logs = {}
        
        for epoch in range(NB_EPOCHS):
            current_training_loss   = self.compute_training_loss()
            current_validation_loss = self.compute_validation_loss()

            logs['loss'] = current_training_loss
            logs['val_loss'] = current_validation_loss

            self.liveloss.update(logs)
            self.liveloss.send()
            
            if min_validation_loss > current_validation_loss:
                print(f'Validation Loss Decreased({min_validation_loss:.6f}--->{current_validation_loss:.6f}) \t Saving The Model')
                min_validation_loss = current_validation_loss
                
                # Saving State Dict
                torch.save(self.model.state_dict(), 'saved_model.pth')

    
    def compute_training_loss(self):
        self.model.train()

        training_loss = 0.0
        for images, labels in self.training_loader:
            images, labels = images.device(self.device), labels.device(self.device)
            
            # Clear the gradients
            self.optimizer.zero_grad()
            # Forward Pass
            target = self.model(images)
            # Find the Loss
            loss = self.loss_function(target, labels)
            # Calculate gradients 
            loss.backward()
            # Update Weights
            self.optimizer.step()
            # Calculate Loss
            training_loss += loss.item().cpu()

        return training_loss
    

    def compute_validation_loss(self):
        with torch.no_grad():
            self.model.eval() 

            validation_loss = 0.0  
            for images, labels in self.validation_loader:
                images, labels = images.device(self.device), labels.device(self.device)
                
                # Forward Pass
                target = self.model(images)
                # Find the Loss
                loss = self.loss_function(target,labels)
                # Calculate Loss
                validation_loss += loss.item().cpu()

            return validation_loss