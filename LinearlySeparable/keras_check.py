# Verify that keras can interact with the backend

import numpy as np
from keras import backend as kbe
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Test keras -backend interaction
data = kbe.variable(np.random.random((4,2)))    # create a 4x2 tensor of random numbers
zero_data = kbe.zeros_like(data)                # create a 4x2 tensor of zeros
print(kbe.eval(zero_data))                      # evaluate the zero_data and print out the results
