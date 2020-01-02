import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.flappy_bird import GameState


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def train(model, start):
    '''
    model: object of the model
    start: start time (current time)
    '''
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = GameState()
    print(game_state)

    # initialize replay memory
    replay_memory = []

    # initialize action (do nothing)
    action = torch.zeros([model.number_of_actions], dtype=torch.float32) # initialze the action tensor to be of size 2 
    action[0] = 1 # [0, 1] --> do nothing | [1, 0] --> fly up
    image_data, reward, terminal = game_state.frame_step(action) # returns the result of the action taken
    image_data = resize_and_bgr2gray(image_data) # crop the floor and resize image to 84x84
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data)).unsqueeze(0) # define state to be the 3 channel version of the returned image


    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0
    
    #decrease epsilon after every iteration
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    #main infinite loop
    while iteration < model.number_of_iterations:

        #get output from neural network
        output = model(state)[0]

        #initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        #epsilon greedy exploration
        random_action = random.random() <= epsilon # select a random number, and if it is less than epsilon, random_action = 1 otherwise random_action = 0
        # this trades off exploration and exploitation --> as the number of iterations increase, the value of epsilon will decrease, so the probability of choosing a random action decreases

        if random_action:
            print("Performing Random Action!")
        #if not performing a random action, choose the best action given by the model
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]  


        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1 #assign the selected action_index to 1 (get either [0, 1] for do nothing or [1, 0] for fly up)

        



def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'pretrained_model/current_model_2000000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = NeuralNetwork()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model, start)


if __name__ == "__main__":
    main(sys.argv[1])
