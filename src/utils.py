# Utilitary methods for ploting, and for the use of the test.py file are kept here

import numpy as np
import matplotlib.pyplot as plt
import math
from torch import empty

def reshapeLabel(label):
    """
    Reshape 1-D [0,1,...] to 2-D [[1,-1],[-1,1],...].
    """
    n = label.size(0)
    y = empty(n, 2)
    y[:, 0] = 2 * (0.5 - label)
    y[:, 1] = - y[:, 0]
    return y.float()

def generate_disk_dataset(nb_points):
    """
    Inspired by the practical 5, this method generates points uniformly in the unit square, with label 1 if the points are in the disc centered at (0.5,0.5) of radius 1/sqrt(2pi), and 0 otherwise
    """
    input = empty(nb_points,2).uniform_(0,1)
    label = input.sub(0.5).pow(2).sum(1).lt(1./2./math.pi).float()
    target = reshapeLabel(label)
    return input,target


def plot_data(x,y,train = True):
    """ Used to plot the generated data from the circle """
    plt.figure(figsize=(8,8))
    radius=(1/(2*math.pi))**(1/2)
    circle=plt.Circle((0.5,0.5),radius,color='g')
    plt.gca().add_artist(circle)
    for i in range(len(x)):
        if y[i][0].item()==1:
            plt.plot(x[i][0],x[i][1],'bo')
        else:
            plt.plot(x[i][0],x[i][1],'r+')
   
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("x",size = 18)
    plt.ylabel("y",size = 18)
    if train :
        plt.title('Distribution of the train set',size = 20)
        plt.savefig("Train set distribution.jpg")  ;
    else : 
        plt.title('Distribution of the test set',size = 20)
        plt.savefig("Test set distribution.jpg")  ;

    plt.show()



def plot_train_error_log(train_log):
    """
    Ploting the train_error log
    """
    plt.figure(figsize=(10,8))
    plt.plot(range(0,100), np.array(train_log)*100)
    plt.xlim(1,100)
    plt.grid()
    plt.xlabel("Epoch", size=15)
    plt.ylabel("Error Rate(%)", size=15)
    plt.title("Training Error Log", size=20)
    plt.savefig("training error log.jpg") 
    
