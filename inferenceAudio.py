import torch
import torchaudio
from urbansoundDataset import UrbanSoundDataset
from cnn import CNNNetwork
from trainAudioDataset import AUDIO_DIR,ANNOTATIONS_FILE,SAMPLE_RATE,NUM_SAMPLES

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def predict(model, input, target, class_mapping):
    model.eval()  # Call eval to turn on the eval switch , you can call train()
    with torch.no_grad():  # Gradient is only useful when doing training so we remove them
        predictions = model(input)
        # Prediction are Tensor objects, 2D Tensor , Tensor (1,10) -> [ [0.1,0.01,....] ] We are looking for the highest number because thats the prediction
        # 1st is dimensions of the input, so if input is array, we expect 1
        # 2nd is number of classes we're trying to predict , in our cases we have 10 because we have 10 numbers in class mapping
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted,expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("cnn_audio_network.pth")
    cnn.load_state_dict(state_dict)

    # load urban sound validation dataset
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,  # Frame size
        hop_length=512,  # Frame size / 2
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    # Get a sample from the validation dataset for inference
    # input data and what it should be
    input, target = usd[1][0], usd[1][1]  # [batch_size, num_channels, frequency, time] , we need to insert a 4th dim in our 3 dim tensor
    input.unsqueeze_(0)

    # Make an inference
    # class mapping is a way to link the integer output of the model to a value
    # in our case MNIST is just numbers so we map to literal numbers, we could hvae linked to non numbers in other models
    predicted, expected = predict(cnn, input, target, class_mapping)

    print(f"Predicted : '{predicted}', expected: '{expected}'")