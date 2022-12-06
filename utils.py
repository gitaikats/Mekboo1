import numpy as np


def sigmoid(x):
    """
     Parameters
     ----------
     x : np.array input data

     Returns
     -------
     np.array
         sigmoid of the input x

     """
    return 1/(1 + np.exp(-x))


def sigmoid_prime(x):
    """
         Parameters
         ----------
         x : np.array input data

         Returns
         -------
         np.array
             derivative of sigmoid of the input x

    """
    return sigmoid(x) * (1-sigmoid(x))



def random_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of xavier initialized np arrays weight matrices

    """

    x = []
    for i in range(len(sizes)-1):
        x.append(xavier_initialization(sizes[i], sizes[i+1]))
    return x



def zeros_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of zero np arrays weight matrices

    """

    x = []
    for i in range(len(sizes) - 1):
        x.append(np.zeros((sizes[i], sizes[i + 1])))
    return x




def zeros_biases(list):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of zero np arrays bias matrices

    """
    x = []
    for i in range(len(list) ):
        x.append(np.zeros(list[i]))
    return x


def create_batches(data, labels, batch_size):
    """
         Parameters
         ----------
         data : np.array of input data
         labels : np.array of input labels
         batch_size : int size of batch

         Returns
         -------
         list
             list of tuples of (data batch of batch_size, labels batch of batch_size)

    """
    minibatches = []
    for i in range(data.shape[0] // batch_size):
        data_mini = data[i * batch_size:(i + 1) * batch_size]
        label_mini = labels[i * batch_size:(i + 1) * batch_size]
        minibatches.append((data_mini, label_mini))
    if (data.shape[0] % batch_size) != 0:
        minibatches.append((data[batch_size * data.shape[0] // batch_size:], labels[batch_size * data.shape[0] // batch_size:]))

    return minibatches
    # x, y, z = [], [], []
    # counter = 0
    # for i, j in zip(data, labels):
    #     y.append(i)
    #     z.append(j)
    #     counter += 1
    #     if(counter == batch_size):
    #         counter = 0
    #         x.append((y,z))
    #         y = []
    #         z = []
    # if(counter!=0):
    #     x.append((y,z))
    # return x


def add_elementwise(list1, list2):
    """
         Parameters
         ----------
         list1 : np.array of numbers
         list2 : np.array of numbers

         Returns
         -------
         list
             list of sum of each two elements by index
    """
    x = []
    for i, j in zip(list1, list2):
        x.append(i + j)
    return x


def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))
