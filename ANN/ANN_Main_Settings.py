from ANN.preprocessing import construct_preprocessor, standarize, replace_nan, whiten
from ANN.activation_functions import tanh_function, sigmoid_function, elliot_function, softsign_function, symmetric_elliot_function, linear_function
from ANN.learning_algorithms  import resilient_backpropagation, backpropagation
from ANN.cost_functions import sum_squared_error, binary_cross_entropy_cost, hellinger_distance, cross_entropy_cost
from ANN.data_structures import Instance
from ANN.neuralnet import NeuralNet
from ANN.tools import print_test
from ANN.final_printout_analysis import error_calculations, round_results
from sklearn import model_selection #cross_validation
from random import shuffle
import csv
from sklearn import datasets

def filereader_split_Xy(file, size_of_output, train_vs_test):
    with open(file, 'rU') as myFile:
        reader = csv.reader(myFile, delimiter=',', dialect = csv.excel)
        next(reader, None)
        filecollector = []
        for row in reader:
            filecollector.append(row)
    myFile.close()

    cleaned_dataset = []

    for row in filecollector:
        temp_row = []
        for value in row:
            if value != '':
                temp_row.append(float(value))
            else:
                temp_row.append("Unknown")

        cleaned_dataset.append(temp_row)

    X = []
    y = []

    for row in cleaned_dataset:
        y_values = row[0 : size_of_output]
        y.append([y_value for y_value in y_values])
        del row[0 : size_of_output]
        X.append(row)

    return X, y

def interpret_text(text):
    if text == "sum_squared_error":
        return sum_squared_error
    if text == "tanh_function":
        return tanh_function
    if text == "sigmoid_function":
        return sigmoid_function
    if text == "sum_squared_error":
        return sum_squared_error
    if text == "elliot_function":
        return elliot_function
    if text == "binary_cross_entropy_cost":
        return binary_cross_entropy_cost
    if text == "hellinger_distance":
        return hellinger_distance
    if text == "cross_entropy_cost":
        return cross_entropy_cost
    if text == "softsign_function":
        return softsign_function
    if text == "symmetric_elliot_function":
        return symmetric_elliot_function
    if text == "linear_function":
        return linear_function

class NeuralNetwork(object):

    def __init__(self, train_dataset_location, test_dataset_location, size_of_output, num_of_neurons, cost_function, neuron_function, activation_function, training_function, max_iterations, max_error, weight_step_min, weight_step_max, start_step, learn_max, learn_min, learning_rate, momentum_factor, hidden_layer_dropout, input_layer_dropout, save_trained_network, network_name, saved_network_location, use_saved_network, view_NN_training, decimal_rounding_for_prediction):
        self.train_dataset_location = train_dataset_location
        self.size_of_output = size_of_output
        self.num_of_neurons = num_of_neurons
        self.cost_function = interpret_text(cost_function)
        self.neuron_function = interpret_text(neuron_function)
        self.activation_function = interpret_text(activation_function)
        self.training_function = training_function

        # General training function parameters
        self.max_iterations = max_iterations
        self.max_error = max_error
        self.save_trained_network = save_trained_network
        self.network_name = network_name
        self.saved_network_location = saved_network_location
        self.use_saved_network = use_saved_network
        self.view_NN_training = view_NN_training
        self.decimal_rounding_for_prediction = decimal_rounding_for_prediction

        # Backpropagation training parameters
        self.learning_rate = learning_rate
        self.momentum_factor = momentum_factor
        self.hidden_layer_dropout = hidden_layer_dropout
        self.input_layer_dropout = input_layer_dropout

        # Resilient backpropagation training parameters
        self.weight_step_min = weight_step_min
        self.weight_step_max = weight_step_max
        self.start_step = start_step
        self.learn_max = learn_max
        self.learn_min = learn_min

        # MAKE SURE FIRST COLUMN(S) OF DATA FILE ARE THE RESULTS
        split_training_data = filereader_split_Xy(train_dataset_location, size_of_output, "training")
        self.X = split_training_data[0]
        self.y = split_training_data[1]
        self.num_of_features = len(self.X[0])

        split_test_data = filereader_split_Xy(test_dataset_location, size_of_output, "test")
        self.X_test_data = split_test_data[0]
        self.y_test_data = split_test_data[1]

    def prep_data(self, X, y):
        # Training sets
            # Instance( [input signals], [target values] )
        trainX, testX, trainy, testy = model_selection.train_test_split(X, y)
        dataset             = [Instance( Xrow, yrow ) for Xrow, yrow in zip(trainX, trainy)]
        preprocessor        = construct_preprocessor( dataset, [replace_nan, standarize] )
        training_data       = preprocessor( dataset )
        test_data           = [Instance( Xrow, yrow ) for Xrow, yrow in zip(testX, testy)]

        return dataset, preprocessor, training_data, test_data

    def nerual_net_basic_settings(self):

        settings            = {
            # Required settings
            "n_inputs"              : self.num_of_features,       # Number of network input signals
            "layers"                : [ (self.num_of_neurons, self.neuron_function), (self.num_of_neurons, self.neuron_function), (self.size_of_output, self.activation_function) ],
                                                # [ (number_of_neurons, activation_function) ]
                                                # The last pair in the list dictate the number of output signals

            # Optional settings
           "weights_low"           : -0.1,     # Lower bound on the initial weight value
            "weights_high"          : 0.1,      # Upper bound on the initial weight value
        }

        return settings

    def train_resilient_backpropagation(self, network, prep_data):
        resilient_backpropagation(
                network,
                prep_data[2],                                   # specify the training set
                prep_data[3],                                   # specify the test set
                self.cost_function,                             # specify the cost function to calculate error
                ERROR_LIMIT          = self.max_error,          # define an acceptable error limit
                max_iterations      = (self.max_iterations),    # continues until the error limit is reach if this argument is skipped

                # optional parameters
                weight_step_max      = self.weight_step_max,
                weight_step_min      = self.weight_step_min,
                start_step           = self.start_step,
                learn_max            = self.learn_max,
                learn_min            = self.learn_min,
                save_trained_network = self.save_trained_network,    # Whether to write the trained weights to disk
                saved_network_location = self.saved_network_location,
                network_name         = self.network_name
            )

    def train_backpropagation(self, network, prep_data):
        backpropagation(
                network,                                        # the network to train
                prep_data[2],                                   # specify the training set
                prep_data[3],                                   # specify the test set
                self.cost_function,                             # specify the cost function to calculate error
                ERROR_LIMIT          = self.max_error,          # define an acceptable error limit
                max_iterations       = (self.max_iterations),     # continues until the error limit is reach if this argument is skipped

                # optional parameters
                learning_rate        = self.learning_rate,           # learning rate
                momentum_factor      = self.momentum_factor,         # momentum
                input_layer_dropout  = self.input_layer_dropout,     # dropout fraction of the input layer
                hidden_layer_dropout = self.hidden_layer_dropout,    # dropout fraction in all hidden layers
                save_trained_network = self.save_trained_network,    # Whether to write the trained weights to disk
                saved_network_location = self.saved_network_location,
                network_name         = self.network_name
            )

    def train_neural_net(self):
        # Initialize the neural network
        settings = NeuralNetwork.nerual_net_basic_settings(self)

        #read a new network
        network     = NeuralNet( settings )
        prep_data = NeuralNetwork.prep_data(self, self.X, self.y)

        # Perform a numerical gradient check
        network.check_gradient( prep_data[2], self.cost_function )

        # select training function and train
        if self.training_function == "resilient backpropagation":
            self.train_resilient_backpropagation(network, prep_data)
        elif self.training_function == "backpropagation":
            self.train_backpropagation(network, prep_data)

        if self.view_NN_training == True:
            print_test( network, prep_data[2], self.cost_function )

        return network

    def predict_output_values(self, network):

        # If you used a preprocessor during the training phase, the
        # preprocessor must also be used on the data used during prediction.

        combine_lists = list(zip(self.X_test_data, self.y_test_data))

        shuffle(combine_lists)

        self.X_test_data, self.y_test_data = zip(*combine_lists)

        predict_dataset = [
                # Instance( [input values] )
                Instance( Xrow ) for Xrow in self.X_test_data
            ]

        correct_output_values = [( yrow[0 : self.size_of_output] ) for yrow in self.y_test_data]
        print ('')
        print ('these are the correct output values')
        print (correct_output_values)
        print ('')

        original_results = [( yrow[0 : self.size_of_output] ) for yrow in self.y_test_data]

        # preprocess the dataset
        preprocessor    = construct_preprocessor( predict_dataset, [replace_nan, standarize] )
        predict_dataset = preprocessor( predict_dataset )

        # feed the instances to the network
        print ('these are the neural net predictions')
        results =  network.predict( predict_dataset ) # return a 2D NumPy array [n_samples, n_outputs]
        print ([result[0 : self.size_of_output] for result in results])

        rounded_results = round_results(results, self.size_of_output, self.decimal_rounding_for_prediction)

        error_calculations(original_results, rounded_results, self.decimal_rounding_for_prediction)

        return correct_output_values, results, rounded_results

def run_ANN(train_dataset_location, test_dataset_location, size_of_output, decimal_rounding_for_prediction, num_of_neurons, cost_function, neuron_function, activation_function, training_function, max_iterations, max_error, weight_step_min, weight_step_max, learn_max, learn_min, learning_rate, start_step, momentum_factor, hidden_layer_dropout, input_layer_dropout, save_trained_network, network_name, saved_network_location, use_saved_network, view_NN_training):
    NN = NeuralNetwork(train_dataset_location, test_dataset_location, size_of_output, num_of_neurons, cost_function, neuron_function, activation_function, training_function, max_iterations, max_error, weight_step_min, weight_step_max, start_step, learn_max, learn_min, learning_rate, momentum_factor, hidden_layer_dropout, input_layer_dropout, save_trained_network, network_name, saved_network_location, use_saved_network, view_NN_training, decimal_rounding_for_prediction)

    NN.prep_data(NN.X, NN.y)
    trained_network = NN.train_neural_net()
    predictions = NN.predict_output_values(trained_network)
    return predictions

