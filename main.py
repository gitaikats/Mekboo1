import os

from collect import *
from max_functions import *
from time import *
from network import *
from matmul_functions import *
from datetime import datetime
start_time = datetime.now()
x = np.array([[4, 1, 4, 3, 20]
             , [3, 5, 2, 4, 25]
             , [1, 0, 1, 1, 12]
             , [10, 3, 8, 0, 7]])
y = np.array([[2, 1, 5, 1, 20]
             , [2, 1, 5, 1, 20]
             , [2, 1, 5, 1, 20]
             , [2, 1, 5, 1, 20]])
# print(sigmoid(x))
# print(sigmoid_prime(x))
# print(random_weights(x))
for i in range(100000):
    max_numba(x, y)

# print(max_numba(x, y))
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

#
# #os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda-9.0/nvvm/lib64/libnvvm.so'
# #os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda-9.0/nvvm/libdevice/'
#
# #initialize layer sizes as list
# layers = [784,128,64,10]
#
# #initialize learning rate
# learning_rate = 0.1
#
# #initialize mini batch size
# mini_batch_size = 16
#
# #initialize epoch
#
# epochs = 5
#
# # initialize training, validation and testing data
# training_data, validation_data, test_data = load_mnist()
#
# start1 = time()
#
# #initialize neuralnet
# nn = NeuralNetwork(layers, learning_rate, mini_batch_size, epochs)
#
# #training neural network
# nn.fit(training_data, validation_data)
#
# stop1 = time()
#
# print('Time matmul_np:', stop1 - start1)
#
# ''' Part 3
#     add training of nn with matmul_numba and matmul_gpu after implementing them
# '''
# #testing neural network
# accuracy = nn.validate(test_data) / 100.0
# print("Test Accuracy: " + str(accuracy) + "%")
