import ANN.ANN_Main_Settings as ANN_Main


# make sure files are in .csv format
# the FIRST COLUMN should be the RESULT / DATA TO PREDICT
train_dataset_location = '/filepath/.../train_dataset.csv'
test_dataset_location = '/filepath/.../test_dataset.csv'

#CHANGE THESE NUMBERS/FUNCTIONS TO ADJUST THE NEURAL NET
# neural net settings
# number of values to be predicted for each row
size_of_output = 1
# number of decimals predictions should be rounded to
decimal_rounding_for_prediction = 1

# number of neurons within each layer (should be about 1/2 the number of features)
num_of_neurons = 10

# sum_squared_error, binary_cross_entropy_cost, hellinger_distance, cross_entropy_cost
cost_function = "sum_squared_error"

# (binary outputs) tanh_function, sigmoid_function, elliot_function, softsign_function, symmetric_elliot_function, softplus_function,
neuron_function = "elliot_function"

# (binary outputs) tanh_function, sigmoid_function, elliot_function, softsign_function, symmetric_elliot_function
# (for non binary outputs) linear_function
activation_function = "linear_function"

# number of training iterations OR train until error rate (whichever  comes first)
# set max_iterations to '' to only train until error rate
max_iterations = 1000
max_error = 1e-3

# resilient backpropagation or backpropagation
training_function = "backpropagation"

# parameters for resilient backpropogation
weight_step_min = 0.0
weight_step_max = 50
learn_min = 0.25
learn_max = 1.5
start_step = 0.5

# parameters for backpropogation
learning_rate = 0.50
momentum_factor = 0.2
hidden_layer_dropout = 0.1
input_layer_dropout = 0.0

# save network or not
save_trained_network = False
saved_network_name = "NN Name"
saved_network_location = "/filepath/.../foldername"

# see the neural net train, not recommended for large datasets
view_NN_training = False

# change to 'True' and include location of saved .pkl file to use previous NN
use_saved_network = [False, '/filepath/.../saved_network.pkl']

# run the neural net
# returns correct_output_values, results, rounded_results
ANN_results = ANN_Main.run_ANN(train_dataset_location, test_dataset_location, size_of_output, decimal_rounding_for_prediction, num_of_neurons, cost_function, neuron_function, activation_function, training_function, max_iterations, max_error, weight_step_min, weight_step_max, learn_max, learn_min, learning_rate, start_step, momentum_factor, hidden_layer_dropout, input_layer_dropout, save_trained_network, saved_network_name, saved_network_location, use_saved_network, view_NN_training)
