
#----------------------------------------------------------
#MODEL

import torch.nn as nn
import torchvision
import torch
import numpy as np
from torch.optim import AdamW 
import copy


class CNNPortion(nn.Module):
    def __init__(self):
        super().__init__()
        layers=nn.ModuleList()

        
        layers.append(nn.Conv2d(in_channels=1, out_channels=12, kernel_size=13, stride=1))
        layers.append(nn.MaxPool2d(2,stride=2)) #first arg is window size
        layers.append(nn.Conv2d(12,18,kernel_size=7,stride=1))
        layers.append(nn.MaxPool2d(2,stride=2)) #first arg is window size
        layers.append(nn.Flatten(1,-1)) #args: start_dim, end_dim. Flatten everything
        layers.append(nn.Linear(72, 8))

        self.model=layers

    def forward(self, inp):
        x=inp
        for lyr in self.model:
            x=lyr(x)
        return x

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        layers=nn.ModuleList()
        
        layers.append(CNNPortion())
        layers.append(CNNPortion())
        layers.append(CNNPortion())
        self.layers=layers 
    
        self.final_layer=nn.Linear(24,5)
    
    def forward(self, front, right, left):
        front_op=self.layers[0](front)
        right_op=self.layers[1](right)
        left_op=self.layers[2](left)
        
        input_final=torch.cat([front_op, right_op, left_op], dim=-1)
        return self.final_layer(input_final)



#RL Part
main=Predictor()
target=Predictor()
optimizer_main=AdamW(main.parameters(),lr=1e-4)
optimizer_target=AdamW(target.parameters(), lr=1e-4)
alpha=0.7
gamma=0.7
r=1 #change this
n=50

#To create tensors with torch.tensor, remember that requires grad=false by default!!! 
#That's why the error occurs when doing loss.backward bc the grad capability isn't enabled!
for step in range(500):
    #1) Predict the Action using the current state (front, right, left)
    front=torch.rand((4,1,32,32))
    left=torch.rand((4,1,32,32))
    right=torch.rand((4,1,32,32)) #REPLACE THESE
    
    x=main(front,left,right)
    Q, action=torch.max(x, dim=-1)
    
    
    #2) Get the s+1 state (where the front, right, left)
    front_prime=torch.rand((4,1,32,32))
    left_prime=torch.rand((4,1,32,32))
    right_prime=torch.rand((4,1,32,32)) #REPLACE THESE
    
    
    #3) Get the Q' value
    Max_Q_Prime,_=torch.max(target(front_prime, left_prime, right_prime), axis=-1)
    
    
    #4) Update the value of the main network
    temp_difference_loss=torch.sum(((alpha * (r * gamma*Max_Q_Prime)) - Q) ** 2)
    temp_difference_loss.backward()
    optimizer_main.step()

    #5) After n steps, update the target network
    if step%50==0:
        target_dict=target.state_dict()
        ref_dict=main.state_dict()
        for key in target_dict.keys():
            target_dict[key]=copy.deepcopy(ref_dict[key])
        target.load_state_dict(target_dict)
        
        opt_dict=optimizer_target.state_dict()
        ref_opt_dict=optimizer_main.state_dict()
        for key in opt_dict.keys():
            opt_dict[key]=copy.deepcopy(ref_opt_dict[key])
        optimizer_target.load_state_dict(opt_dict)
        
        




