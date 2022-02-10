from time import sleep

from tqdm import tqdm
from RAVE.common.Trainer import Trainer


class AudioTrainer(Trainer):
    def __init__(self, training_loader, validation_loader, loss_function, device, model, optimizer, scheduler, ROOT_DIR_PATH, CONTINUE_TRAINING):
        super().__init__(training_loader, validation_loader, loss_function, device, model, optimizer, scheduler, ROOT_DIR_PATH, CONTINUE_TRAINING)
    
    def compute_training_loss(self):
        self.model.train()

        training_loss = 0.0
        number_of_wavs =0

        for audios, labels in tqdm(self.training_loader, 'training', leave=False):
            sleep(0)
        return 0

    def compute_validation_loss(self):
        return 0