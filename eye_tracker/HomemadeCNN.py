from EyeTrackerDataset import IMAGE_DIMENSIONS

from torch import nn
import numpy as np

class HomemadeCNN(nn.Module):
    def __init__(self):
        super(HomemadeCNN, self).__init__()

        self.current_image_size = np.array(IMAGE_DIMENSIONS[1:3], dtype=int)
        self.current_number_of_channels = IMAGE_DIMENSIONS[0]

        self.conv_block_1 = self.get_convolutionnal_block(9, 64)
        self.conv_block_2 = self.get_convolutionnal_block(5, 128)
        self.conv_block_3 = self.get_convolutionnal_block(3, 256)
        self.conv_block_4 = self.get_convolutionnal_block(3, 512)
        self.conv_block_5 = self.get_convolutionnal_block(3, 1024)

        self.fully_connected_layers = nn.Sequential()
        self.fully_connected_layers.add_module('fc1', nn.Linear(int(self.current_image_size[0] * self.current_image_size[1] * self.current_number_of_channels), 512) )
        self.fully_connected_layers.add_module('relu1', nn.ReLU() )
        self.fully_connected_layers.add_module('dropout1', nn.Dropout() )
        
        self.fully_connected_layers.add_module('fc2', nn.Linear(512, 256) )
        self.fully_connected_layers.add_module('relu2', nn.ReLU() )
        self.fully_connected_layers.add_module('dropout2', nn.Dropout() )

        self.fully_connected_layers.add_module('fc3', nn.Linear(256, 128) )
        self.fully_connected_layers.add_module('relu3', nn.ReLU() )
        self.fully_connected_layers.add_module('dropout3', nn.Dropout() )

        self.fully_connected_layers.add_module('out', nn.Linear(128, 5) )


    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)

        x = x.view(x.size()[0], -1) # Flatten
        x = self.fully_connected_layers(x)

        return x 


    def get_convolutionnal_block(self, kernel_size, out_number_of_channels, stride=1, dilatation=1):
        conv_block = nn.Sequential()

        same_padding = HomemadeCNN.get_same_padding(self.current_image_size, kernel_size, stride, dilatation)
        conv_block.add_module('conv_same_padding_1', nn.Conv2d(self.current_number_of_channels, self.current_number_of_channels, kernel_size, stride, same_padding, dilatation))
        conv_block.add_module('relu_1', nn.ReLU())
        conv_block.add_module('dropout_1', nn.Dropout(0.05))

        conv_block.add_module('conv_same_padding_2', nn.Conv2d(self.current_number_of_channels, out_number_of_channels, kernel_size, stride, same_padding, dilatation))
        self.current_number_of_channels = out_number_of_channels

        padding = HomemadeCNN.get_padding(self.current_image_size, self.current_image_size//2, kernel_size, 2, dilatation)
        conv_block.add_module('conv_stride_of_2', nn.Conv2d(self.current_number_of_channels, self.current_number_of_channels, kernel_size, 2, padding, dilatation))
        self.current_image_size = self.current_image_size//2 

        conv_block.add_module('relu_2', nn.ReLU())
        conv_block.add_module('dropout_2', nn.Dropout(0.05))

        return conv_block

    
    def get_same_padding(input_dimensions, kernel_size, stride, dilatation):
        return HomemadeCNN.get_padding(input_dimensions, input_dimensions, kernel_size, stride, dilatation)


    def get_padding(input_dimensions, output_dimensions, kernel_size, stride, dilatation):
        height_in, width_in = input_dimensions
        height_out, width_out = output_dimensions

        height_padding = ( (height_out - 1)*stride - height_in + dilatation*(kernel_size-1)+2 )//2 
        width_padding = ( (width_out - 1)*stride   - width_in  + dilatation*(kernel_size-1)+2 )//2 

        return ( int(height_padding), int(width_padding) )