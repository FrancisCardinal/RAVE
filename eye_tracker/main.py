import torch

from EyeTrackerModel import EyeTrackerModel
from ellipse_util import ellipse_loss_function

from Trainer import Trainer
from EyeTrackerDataset import EyeTrackerDataset

def main():
    DEVICE = 'cpu'
    if( torch.cuda.is_available() ): 
        DEVICE = 'cuda'

    BATCH_SIZE = 128 
    training_sub_dataset, validation_sub_dataset = EyeTrackerDataset.get_training_and_validation_sub_datasets()

    training_loader = torch.utils.data.DataLoader(training_sub_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers=8, pin_memory=True, persistent_workers=True )

    validation_loader = torch.utils.data.DataLoader(validation_sub_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers=8, pin_memory=True, persistent_workers=True )

    eye_tracker_model = EyeTrackerModel()
    eye_tracker_model.to(DEVICE)
    print(eye_tracker_model)
    optimizer = torch.optim.SGD(eye_tracker_model.parameters(), lr=0.01, momentum=0.9)

    trainer = Trainer(training_loader, 
                      validation_loader, 
                      ellipse_loss_function,
                      DEVICE,
                      eye_tracker_model,
                      optimizer)
    
    trainer.train_with_validation()

    test_sub_dataset = EyeTrackerDataset.get_test_sub_dataset
    test_loader = torch.utils.data.DataLoader(test_sub_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=8, pin_memory=True, persistent_workers=True )
if __name__ =='__main__':
    main()