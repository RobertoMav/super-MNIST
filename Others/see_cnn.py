from manim import *

from manim_ml.neural_network import Convolutional2DLayer, FeedForwardLayer, NeuralNetwork, MaxPooling2DLayer

# This changes the resolution of our rendered videos
config.pixel_height = 700
config.pixel_width = 1900
config.frame_height = 7.0
config.frame_width = 7.0

# Here we define our basic scene
class BasicScene(ThreeDScene):

    # The code for generating our scene goes here
    def construct(self):    
        # Make the neural network
        nn = NeuralNetwork([
                Convolutional2DLayer(1, 6, 3, filter_spacing=0.32),
                MaxPooling2DLayer(kernel_size=2),
                Convolutional2DLayer(6, 16, 3, filter_spacing=0.32),
                MaxPooling2DLayer(kernel_size=2),
                #FeedForwardLayer(400, activation_function="ReLU"),
                #FeedForwardLayer(512, activation_function="ReLU"),
                FeedForwardLayer(10)
            ],
            layer_spacing=0.25,
        )
        # Center the neural network
        nn.move_to(ORIGIN)
        self.add(nn)
        # Make a forward pass animation
        forward_pass = nn.make_forward_pass_animation()
        # Play animation
        self.play(forward_pass)