import math
import Module as m
import utils as u
from torch import empty
from torch import set_grad_enabled
set_grad_enabled(False)

# Generate train set and test set
nb_points = 1000
train_input, train_target = u.generate_disk_dataset(nb_points)
test_input,test_target = u.generate_disk_dataset(nb_points)

# Normalization
mu, std = train_input.mean(), train_input.std()
train_input.sub_(mu).div_(std)
test_input.sub_(mu).div_(std)

# Build the model

model = m.Sequential()
model.build(m.Linear(2,25),m.ReLU(),m.Linear(25,25),m.ReLU(),m.Linear(25,25),m.ReLU(),m.Linear(25,2),m.Tanh())

# Set parameters of the neural network optimization procedure 

epochs = 100
learning_rate = 5e-1
mini_batch_size = 50
criterion = m.LossMSE() 

# Train the model and logging the error
f = open("error_logs.txt","w")
train_log = []
for epoch in range(epochs):
    f.write("\n-------------Starting epoch {}------------- : \n".format(epoch))
    for batch in range(0,train_input.size(0),mini_batch_size):
        output = model.forward(train_input.narrow(0,batch,mini_batch_size))
        loss = criterion.forward(output,train_target.narrow(0,batch,mini_batch_size).float())
    
        error = (output.max(1)[1].ne(train_target.narrow(0, batch, mini_batch_size).max(1)[1]).sum()).item()/output.size(0)
        f.write("Epoch {} Batch {} Loss {:4.2f} error {:4.2f}% \n".format(epoch,int(batch/mini_batch_size),loss,error*100))
        gradient = criterion.backward()
        model.backward(gradient)
        
        #update parameters : p = p - learning_rate * dL/dp
        for params in model.param():
            for param in params : 
                param[0].sub_(learning_rate * param[1])
                
                
    f.write("\n------------- After Epoch {} ------------- \n \n".format(epoch))
    output = model.forward(train_input)
    loss = criterion.forward(output, train_target.float())
    error = output.max(1)[1].ne(train_target.max(1)[1]).sum().item()/output.size(0)
    f.write("Loss: {:4.2f}, Error: {:6.2%}\n".format(loss, error))
    train_log.append(error)        
        


# Printing the errors 

train_output = model.forward(train_input)
train_error = (train_output.max(1)[1].ne(train_target.max(1)[1]).sum()).item()/train_output.size(0)
print("Final Train Error: {:.2%} \n".format(train_error))
f.write("Final Train Error: {:.2%} \n".format(train_error))

test_output = model.forward(test_input)
test_error = (test_output.max(1)[1].ne(test_target.max(1)[1]).sum()).item()/test_output.size(0)
print("Final Test Error: {:.2%} \n".format(test_error))      
f.write("Final Test Error: {:.2%} \n".format(test_error))
f.close()



     
