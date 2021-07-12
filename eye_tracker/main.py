import torch
import cv2
import numpy as np
from time import sleep

from EyeTrackerModel import EyeTrackerModel
from ellipse_util import ellipse_loss_function, draw_ellipse_on_image

from Trainer import Trainer
from EyeTrackerDataset import EyeTrackerDataset

def main():
    DEVICE = 'cpu'
    if( torch.cuda.is_available() ): 
        DEVICE = 'cuda'

    BATCH_SIZE = 128 
    training_sub_dataset, validation_sub_dataset = EyeTrackerDataset.get_training_and_validation_sub_datasets(EyeTrackerDataset.get_transform())

    training_loader = torch.utils.data.DataLoader(training_sub_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers=8, pin_memory=True, persistent_workers=True )

    validation_loader = torch.utils.data.DataLoader(validation_sub_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers=8, pin_memory=True, persistent_workers=True )

    eye_tracker_model = EyeTrackerModel()
    eye_tracker_model.to(DEVICE)
    print(eye_tracker_model)
    optimizer = torch.optim.SGD(eye_tracker_model.parameters(), lr=0.001, momentum=0.9)

    trainer = Trainer(training_loader, 
                      validation_loader, 
                      ellipse_loss_function,
                      DEVICE,
                      eye_tracker_model,
                      optimizer)
    
    trainer.train_with_validation()

    Trainer.load_best_model(eye_tracker_model)

    visualize_predictions(eye_tracker_model, validation_loader, DEVICE)

    test_sub_dataset = EyeTrackerDataset.get_test_sub_dataset(EyeTrackerDataset.get_transform())
    test_loader = torch.utils.data.DataLoader(test_sub_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=8, pin_memory=True, persistent_workers=True )

def visualize_predictions(model, data_loader, DEVICE):
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            predictions = model(images)
            for image, prediction, label in zip(images, predictions, labels):
                image = image.permute(1, 2, 0).cpu().numpy()
                image *= 255.0
                image = np.ascontiguousarray(image, dtype=np.uint8)

                image = draw_ellipse_on_image(image, prediction, color=(255, 0, 0))
                image = draw_ellipse_on_image(image, label,  color=(0, 255, 0))

                cv2.imshow('validation', image)
                cv2.waitKey(1500)

if __name__ =='__main__':
    main()