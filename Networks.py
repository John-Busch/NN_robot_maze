import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    #__init__() initiatizes the nn.Module
    def __init__(self):
        pass

    #forward pass through the network. output needs to be a tensor
    def forward(self, input):
        return output

    #return the loss over testing dataset. Pytorch loss function. Must return a value. 
    def evaluate(self, model, test_loader, loss_function):
        return loss

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
