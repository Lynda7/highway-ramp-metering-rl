from __future__ import absolute_import 
from __future__ import print_function
"""
These imports include useful libraries for managing the simulation (SUMO and Traci),
machine learning (PyTorch, NumPy), and graphical display (matplotlib).

"""
import os
import sys
import optparse
import random
import serial  # for serial communication with external devices
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from function.plot_function import plot_graphs

"""================================================================================
=============================Imporatation of  Control Functions======================================================
==================================================================================="""
from function.SUMO_setup import setup_sumo_environment
from function.Contol_functions import count_vehicles_in_lanes,calculate_total_wait_time
from function.Contol_functions import set_traffic_phase_duration ,get_command_line_options
from sumolib import checkBinary  
import traci

"""=========================================================================================================
====================SUMO Environment Configuration=========================================================
=============================================================================================================================================="""

setup_sumo_environment()

"""===============================================================================================================================
============================Neural Network Model Class===================================================================
=============================================================================================================================="""


class NeuralNetworkModel(nn.Module): 


    def __init__(self, learning_rate, input_dims, hidden_layer1, hidden_layer2, num_actions):
        """
        A neural network model used to approximate Q-values in reinforcement learning.

        Parameters:
        ----------
        learning_rate : float
            Learning rate used for the optimizer.

        input_dims : int
            Number of input dimensions (features) for the neural network.

        hidden_layer1 : int
            Number of neurons in the first hidden layer of the neural network.

        hidden_layer2 : int
            Number of neurons in the second hidden layer of the neural network.

        num_actions : int
            Number of output actions, corresponding to the number of possible actions in the environment.

        Attributes:
        ----------
        fc1 : nn.Linear
            First fully connected layer that takes the input dimensions and maps it to the first hidden layer.

        fc2 : nn.Linear
            Second fully connected layer that maps from the first hidden layer to the second hidden layer.

        fc3 : nn.Linear
            Output fully connected layer that maps from the second hidden layer to the output actions.

        optimizer : torch.optim.Adam
            Adam optimizer used for updating the model's weights based on the computed loss.

        loss : nn.MSELoss
            Loss function used to calculate the mean squared error between the predicted and target Q-values.

        device : torch.device
            The device (GPU or CPU) where the model is loaded.

        Methods:
        -------
        forward(state):
            Passes the input `state` through the network layers to compute the predicted Q-values for each action.
        """
        super(NeuralNetworkModel, self).__init__()
        self.lr = learning_rate
        self.input_dims = input_dims
        self.hidden_layer1 = hidden_layer1
        self.hidden_layer2 = hidden_layer2
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.input_dims, self.hidden_layer1)
        self.fc2 = nn.Linear(self.hidden_layer1, self.hidden_layer2)
        self.fc3 = nn.Linear(self.hidden_layer2, self.num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))  
        x = torch.relu(self.fc2(x))      
        actions = self.fc3(x)
        return actions
    



"""===============================================================================================
========================Agent Class==========================================================
==================================================================================================="""

class TrafficAgent:



    """
    An agent that implements the Q-learning algorithm for reinforcement learning using deep learning models.

    Parameters:
    ----------
    gamma : float
        The discount factor used to calculate the future rewards.

    epsilon : float
        The initial exploration rate used to balance exploration and exploitation.

    learning_rate : float
        The learning rate used for model optimization.

    input_dims : int
        The number of dimensions in the input state space.

    hidden_layer1 : int
        The number of neurons in the first hidden layer of the neural network.

    hidden_layer2 : int
        The number of neurons in the second hidden layer of the neural network.

    batch_size : int
        The number of experiences used in each mini-batch during training.

    num_actions : int
        The total number of possible actions the agent can take.

    junctions : list
        The list of junctions where the agent operates, which influences memory structure.

    max_memory_size : int, optional
        The maximum size of the experience replay memory. Default is 100,000.

    epsilon_decrease : float, optional
        The rate at which the exploration rate decreases. Default is 5e-4.

    epsilon_min : float, optional
        The minimum value the exploration rate can reach. Default is 0.05.

    Attributes:
    ----------
    gamma : float
        The discount factor used in Q-learning to weigh future rewards.

    epsilon : float
        The current exploration rate that decreases over time.

    learning_rate : float
        The learning rate used to optimize the model.

    batch_size : int
        The number of transitions to use in each batch for training.

    input_dims : int
        The dimensionality of the input state space.

    hidden_layer1 : int
        The number of neurons in the first hidden layer.

    hidden_layer2 : int
        The number of neurons in the second hidden layer.

    num_actions : int
        The number of actions the agent can take.

    model :
        A neural network model used to approximate the Q-values for each action.

    memory : dict
        A dictionary storing the agent's experience replay memory for each junction, including state, reward, action, etc.

    memory_counter : int
        A counter for the number of transitions stored in memory.

    iteration_counter : int
        A counter for the number of iterations during training.

    replace_target : int
        The number of iterations before replacing the target model (for Double DQN).

    Methods:
    -------
    store_transition(state, new_state, action, reward, done, junction):
        Stores the agent's experience for a given junction in memory.

    choose_action(observation):
        Chooses an action based on the current state (observation), using an epsilon-greedy policy.

    reset(junction_numbers):
        Resets the memory of specified junctions.

    save(model_name):
        Saves the trained model weights to a file.

    learn(junction):
        Performs a training step using experience replay and Q-learning updates to the Q-network.
    """


    def __init__(
        self,
        gamma, 
        epsilon, 
        learning_rate, 
        input_dims, 
        hidden_layer1, 
        hidden_layer2, 
        batch_size,
        num_actions,
        junctions, 
        max_memory_size=100000,
        epsilon_decrease=5e-4, 
        epsilon_min=0.05, 
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.hidden_layer1 = hidden_layer1
        self.hidden_layer2 = hidden_layer2
        self.num_actions = num_actions
        self.action_space = [i for i in range(num_actions)]
        self.junctions = junctions
        self.max_memory_size = max_memory_size
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_min = epsilon_min
        self.memory_counter = 0
        self.iteration_counter = 0
        self.replace_target = 100

        self.model = NeuralNetworkModel(self.learning_rate, self.input_dims, self.hidden_layer1, self.hidden_layer2, self.num_actions)
        self.memory = {junction: {
                "state_memory": np.zeros((self.max_memory_size, self.input_dims), dtype=np.float32),
                "new_state_memory": np.zeros((self.max_memory_size, self.input_dims), dtype=np.float32),
                "reward_memory": np.zeros(self.max_memory_size, dtype=np.float32),
                "action_memory": np.zeros(self.max_memory_size, dtype=np.int32),
                "terminal_memory": np.zeros(self.max_memory_size, dtype=bool),
                "memory_counter": 0,
                "iteration_counter": 0,
            } for junction in junctions
        }
    def store_transition(self, state, new_state, action, reward, done, junction):
        
        if junction not in self.memory:
            
            self.memory[junction] = {
                "state_memory": np.zeros((self.max_memory_size, self.input_dims), dtype=np.float32),
                "new_state_memory": np.zeros((self.max_memory_size, self.input_dims), dtype=np.float32),
                "reward_memory": np.zeros(self.max_memory_size, dtype=np.float32),
                "action_memory": np.zeros(self.max_memory_size, dtype=np.int32),
                "terminal_memory": np.zeros(self.max_memory_size, dtype=bool),
                "memory_counter": 0,
                "iteration_counter": 0,
            }

        
        index = self.memory[junction]["memory_counter"] % self.max_memory_size
        self.memory[junction]["state_memory"][index] = state
        self.memory[junction]["new_state_memory"][index] = new_state
        self.memory[junction]['reward_memory'][index] = reward
        self.memory[junction]['terminal_memory'][index] = done
        self.memory[junction]["action_memory"][index] = action
        self.memory[junction]["memory_counter"] += 1


    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.model.device)
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.model.forward(state)
            action = torch.argmax(actions).item()
        return action

    def reset(self, junction_numbers):
        for junction_number in junction_numbers:
            self.memory[junction_number]['memory_counter'] = 0

    def save(self, model_name):
        torch.save(self.model.state_dict(), f'traffic_control_models/{model_name}.bin')

    def learn(self, junction):
        self.model.optimizer.zero_grad()
        
        batch_indices = np.arange(self.memory[junction]['memory_counter'], dtype=np.int32)
        
        state_batch = torch.tensor(self.memory[junction]["state_memory"][batch_indices]).to(self.model.device)
        new_state_batch = torch.tensor(self.memory[junction]["new_state_memory"][batch_indices]).to(self.model.device)
        reward_batch = torch.tensor(self.memory[junction]['reward_memory'][batch_indices]).to(self.model.device)
        terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch_indices]).to(self.model.device)
        action_batch = self.memory[junction]["action_memory"][batch_indices]

        q_eval = self.model.forward(state_batch)[batch_indices, action_batch]
        q_next = self.model.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        
        loss = self.model.loss(q_target, q_eval).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()

        self.iteration_counter += 1
        self.epsilon = max(self.epsilon - self.epsilon_decrease, self.epsilon_min)



"""=============================================================================================
=======================Simulation Function=======================================================
======================================================================================================"""



def run(is_training=True, model_filename="model", total_epochs=30, total_steps=300):

    """
    Runs a traffic management simulation using SUMO and a reinforcement learning agent.

    Parameters:
    ----------
    is_training : bool, optional
        Indicates whether the simulation should run in training mode (True) or evaluation mode (False).
        Default is True.

    model : str, optional
        Name of the file used to save or load the trained model.
        Default is "model".

    epochs : int, optional
        Total number of epochs to run during training.
        An epoch corresponds to one complete execution of the simulation.
        Default is 30.

    total_steps : int, optional
        Total number of simulation steps to execute in each epoch.
        Default is 300.

    Description:
    -----------
    The function starts a SUMO simulation, initializes a reinforcement learning agent, and manages
    traffic lights in a simulated network. It either trains the model (in training mode) or evaluates
    its performance (in evaluation mode). Results include the total vehicle waiting time and rewards
    obtained. In training mode, performance results are logged and saved as plots.

    Returns:
    --------
    None
        Outputs are displayed and saved in specific files (model and visualization plots).
    """
    
    best_time = np.inf  
    total_waiting_times = []  
    total_rewards = [] 
    

    traci.start(
        [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "network_data/tripinfo.xml"]
    )

    
    junctions = traci.trafficlight.getIDList()
    junction_indices = list(range(len(junctions)))

    
    agent = TrafficAgent(
        gamma=0.99,
        epsilon=0.0,
        learning_rate=0.1,
        input_dims=4,
        hidden_layer1=256,
        hidden_layer2=256,
        batch_size=1024,
        num_actions=4,
        junctions=junctions,
    )

    if not is_training:  
        agent.model.load_state_dict(torch.load(f'traffic_control_models/{model_filename}.bin', map_location=agent.model.device),strict=False)

    print(f"Using device: {agent.model.device}")
    traci.close()  

    for epoch in range(total_epochs):
        if is_training:  
            traci.start(
                [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
            )
        else:
            traci.start(
                [checkBinary("sumo-gui"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
            )

        print(f"Epoch: {epoch}") 

        
        phase_transitions = [
            ["rggg", "yggg"],
            ["yggg", "rggg"],
            ["gggg", "yggg"],
            ["yggg", "gggg"],
        ]
        
  
        step = 0
        total_time = 0
        min_duration = 5
        traffic_light_times = {}
        prev_waiting_times = {}
        prev_vehicles_per_lane = {}
        previous_action = {}
        all_lanes = []


        for idx, junction in enumerate(junctions):
            prev_waiting_times[junction] = 0
            previous_action[idx] = 0
            traffic_light_times[junction] = 0
            prev_vehicles_per_lane[idx] = [0] * 4
            all_lanes.extend(list(traci.trafficlight.getControlledLanes(junction)))

        while step <= total_steps:  
            traci.simulationStep() 
            
            for idx, junction in enumerate(junctions):  
                controlled_lanes = traci.trafficlight.getControlledLanes(junction)
                waiting_time = calculate_total_wait_time(controlled_lanes)  
                total_time += waiting_time  
                
                if traffic_light_times[junction] == 0:  
                    vehicles_per_lane = count_vehicles_in_lanes(controlled_lanes)

                    reward = -waiting_time  
                    
                    current_state = list(vehicles_per_lane.values()) 
                    previous_state = prev_vehicles_per_lane[idx]
                    prev_vehicles_per_lane[idx] = current_state
                    
                    agent.store_transition(previous_state, current_state, previous_action[idx], reward, (step == total_steps), idx)
                    
                    action = agent.choose_action(current_state)
                    previous_action[idx] = action
                    
                    set_traffic_phase_duration(junction, 6, phase_transitions[action][0])
                    set_traffic_phase_duration(junction, min_duration + 10, phase_transitions[action][1])
                    traffic_light_times[junction] = min_duration + 10
                    
                    if is_training:  
                        agent.learn(idx)
                else:  
                    traffic_light_times[junction] -= 1

            step += 1  

        
        print(f"Total waiting time: {total_time}")  # Afficher le temps d'attente total pour cette époque
        total_waiting_times.append(total_time)
        total_rewards.append(reward)

        if total_time < best_time:  
            best_time = total_time
            if is_training:
                agent.save(model_filename)

        traci.close()  
        sys.stdout.flush()  
        if not is_training:  
            break

    if is_training: 

          plot_graphs(total_waiting_times, total_rewards, model_filename) 


"""
=================================================================================================
==============================    main   ===========================================================
==============================================================================================
"""

if __name__ == "__main__":
    options = get_command_line_options()
    model_filename = options.model_name
    is_training = options.train
    total_epochs = options.epochs
    total_steps = options.steps
    run(is_training=is_training, model_filename=model_filename, total_epochs=total_epochs, total_steps=total_steps)
