from Layer import Dense,Activation,Tanh,mse,mse_prime
import numpy as np
import matplotlib.pyplot as plt

X = np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))
Y = np.reshape([[0],[1],[1],[1]],(4,1,1))

network1 = [ Dense(2,4),Tanh(),
           Dense(4,5),Tanh(),Dense(5,1),
           Tanh()]

network2 = [ Dense(2,3),Tanh(),
           Dense(3,1),
           Tanh()]

net = [network1,network2]

epoch = 80000
learning_rate_list = [0.1,0.2]

learning_rate = learning_rate_list[0]

result_list_1 = []
result_list_2 = []

index = 0

for network in net:
    index +=1

    for e in range(epoch):
        error = 0
        for x,y in zip(X,Y):
            #forward+
            output = x
            for layer in network:
                output = layer.forward(output)

            error += mse(y,output)

            #backward 
            grad = mse_prime(y,output)
            for layer in reversed(network):
                grad = layer.backward(grad,learning_rate)
                


        error /=len(X)
        if (e%10 == 0):
            match(index):
                case 1:
                    result_list_1.append(error)
                case 2:
                    result_list_2.append(error)    
                
        #print(f"{e+1}/{epoch} error = {error}")        

#res = [result_list_1[x] -result_list_2[x]  for x in range(len(result_list_1))]
    
plt.figure()
plt.xlabel("Nombre d'epoch")
plt.title("Erreur en fonction du nombre d'epoch")
plt.plot(result_list_1[1000:])
plt.plot(result_list_2[1000:])
plt.grid()
plt.show() 
