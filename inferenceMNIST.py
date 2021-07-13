import torch
from trainMNIST import FeedForwardNet, download_mnist_datasets

class_mapping = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
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
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)

    # load MNIST validation dataset
    _, validation_data = download_mnist_datasets()

    # Get a sample from the validation dataset for inference
    # input data and what it should be
    input, target = validation_data[0][0], validation_data[0][1]

    # Make an inference
    # class mapping is a way to link the integer output of the model to a value
    # in our case MNIST is just numbers so we map to literal numbers, we could hvae linked to non numbers in other models
    predicted, expected = predict(feed_forward_net, input, target, class_mapping)

    print(f"Predicted : '{predicted}', expected: '{expected}'")