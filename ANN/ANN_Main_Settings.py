from preprocessing import construct_preprocessor, standarize, replace_nan, whiten
from activation_functions import tanh_function, sigmoid_function, elliot_function, softsign_function, symmetric_elliot_function, linear_function
from learning_algorithms  import resilient_backpropagation, backpropagation
from cost_functions import sum_squared_error, binary_cross_entropy_cost, hellinger_distance, cross_entropy_cost
from data_structures import Instance
from neuralnet import NeuralNet
from tools import print_test
from sklearn import model_selection #cross_validation
from random import shuffle
import csv
from sklearn import datasets

def filereader_split_Xy(file, size_of_output):
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
            temp_row.append(float(value))
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

    def __init__(self, train_dataset_location, test_dataset_location, size_of_output, num_of_neurons, cost_function, neuron_function, activation_function, training_function, max_iterations, max_error):
        self.train_dataset_location = train_dataset_location
        self.size_of_output = size_of_output
        self.num_of_neurons = num_of_neurons
        self.cost_function = interpret_text(cost_function)
        self.neuron_function = interpret_text(neuron_function)
        self.activation_function = interpret_text(activation_function)
        self.training_function = training_function
        self.max_iterations = max_iterations
        self.max_error = max_error

        # MAKE SURE FIRST COLUMN(S) OF DATA FILE ARE THE RESULTS
        split_file = filereader_split_Xy(train_dataset_location, size_of_output)
        self.X = split_file[0]
        self.y = split_file[1]
        self.num_of_features = len(self.X[0])

        self.X_test_data = filereader_split_Xy(test_dataset_location, size_of_output)[0]
        self.y_test_data = filereader_split_Xy(test_dataset_location, size_of_output)[1]

    def truncate(self, f, n):
        '''Truncates/pads a float f to n decimal places without rounding'''
        s = '{}'.format(f)
        if 'e' in s or 'E' in s:
            return '{0:.{1}f}'.format(f, n)
        i, p, d = s.partition('.')
        return '.'.join([i, (d+'0'*n)[:n]])

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

    def train_and_predict(self, X, y, X_test_data, y_test_data, weight_step_min, weight_step_max, learn_max, learn_min, learning_rate, start_step, momentum_factor, hidden_layer_dropout, input_layer_dropout, save_trained_network, network_name, saved_network_location, use_saved_network, view_NN_training):

        if use_saved_network[0] == True:
            settings = NeuralNetwork.nerual_net_basic_settings(self)

            network = NeuralNet.load_network_from_file( use_saved_network[1] )

            # If you used a preprocessor during the training phase, the
            # preprocessor must also be used on the data used during prediction.

            # MAKE SURE FIRST COLUMN OF DATA FILE IS CONCUSSED (1) OR NOT CONCUSSED (0)

            combine_lists = list(zip(X_test_data, y_test_data))

            shuffle(combine_lists)

            X_test_data, y_test_data = zip(*combine_lists)

            predict_dataset = [
                    # Instance( [input values] )
                    Instance( Xrow ) for Xrow in X_test_data
                ]
            print ''
            print 'these are the original predicted results'
            print [( yrow ) for yrow in y_test_data] #test_set_y[7],test_set_y[65],test_set_y[20]
            print ''

            original_results = [( yrow ) for yrow in y_test_data]

            # preprocess the dataset
            preprocessor    = construct_preprocessor( predict_dataset, [replace_nan, standarize] )
            predict_dataset = preprocessor( predict_dataset )

            # feed the instances to the network
            print 'these are the neural net predictions'
            results =  network.predict( predict_dataset ) # return a 2D NumPy array [n_samples, n_outputs]
            print results

            rounded_results = []
            for value in results:
                rounded_results.append(int(round(value)))

            print ''
            print 'these are the rounded results'
            print rounded_results

            print ''
            accuracy = []
            for (original, NNpred) in zip(original_results, rounded_results):
                if original[0] != NNpred:
                    accuracy.append(0)
                else:
                    accuracy.append(1)

            def truncate(f, n):
                '''Truncates/pads a float f to n decimal places without rounding'''
                s = '{}'.format(f)
                if 'e' in s or 'E' in s:
                    return '{0:.{1}f}'.format(f, n)
                i, p, d = s.partition('.')
                return '.'.join([i, (d+'0'*n)[:n]])

            perc_accuracy = (sum(accuracy)/float(len(accuracy)))*100
            perc_accuracy = truncate(perc_accuracy,1)

            print 'percentage accuracy: ' + str(perc_accuracy) + "%"

        else:
            # Initialize the neural network
            # read old network data
            settings = NeuralNetwork.nerual_net_basic_settings(self)


            #read a new network
            network     = NeuralNet( settings )

            prep_data = NeuralNetwork.prep_data(self, X, y)
            # Perform a numerical gradient check
            network.check_gradient( prep_data[2], self.cost_function )

            def train_resilient_backpropagation():
                resilient_backpropagation(
                        network,
                        prep_data[2],                                   # specify the training set
                        prep_data[3],                                   # specify the test set
                        self.cost_function,                             # specify the cost function to calculate error
                        ERROR_LIMIT          = self.max_error,          # define an acceptable error limit
                        max_iterations      = (self.max_iterations),    # continues until the error limit is reach if this argument is skipped

                        # optional parameters
                        weight_step_max      = weight_step_max,
                        weight_step_min      = weight_step_min,
                        start_step           = start_step,
                        learn_max            = learn_max,
                        learn_min            = learn_min,
                        save_trained_network = save_trained_network,    # Whether to write the trained weights to disk
                        saved_network_location = saved_network_location,
                        network_name         = network_name
                    )

            def train_backpropagation():
                backpropagation(
                        network,                                        # the network to train
                        prep_data[2],                                   # specify the training set
                        prep_data[3],                                   # specify the test set
                        self.cost_function,                             # specify the cost function to calculate error
                        ERROR_LIMIT          = self.max_error,          # define an acceptable error limit
                        max_iterations       = (self.max_iterations),     # continues until the error limit is reach if this argument is skipped

                        # optional parameters
                        learning_rate        = learning_rate,           # learning rate
                        momentum_factor      = momentum_factor,         # momentum
                        input_layer_dropout  = input_layer_dropout,     # dropout fraction of the input layer
                        hidden_layer_dropout = hidden_layer_dropout,    # dropout fraction in all hidden layers
                        save_trained_network = save_trained_network,    # Whether to write the trained weights to disk
                        saved_network_location = saved_network_location,
                        network_name         = network_name
                    )


            if self.training_function == "resilient backpropagation":
                train_resilient_backpropagation()
            elif self.training_function == "backpropagation":
                train_backpropagation()


            if view_NN_training == True:
                print_test( network, prep_data[2], self.cost_function )
            # If you used a preprocessor during the training phase, the
            # preprocessor must also be used on the data used during prediction.

            combine_lists = list(zip(X_test_data, y_test_data))

            shuffle(combine_lists)

            X_test_data, y_test_data = zip(*combine_lists)

            predict_dataset = [
                    # Instance( [input values] )
                    Instance( Xrow ) for Xrow in X_test_data
                ]

            correct_output_values = [( yrow[0 : self.size_of_output] ) for yrow in y_test_data]
            print ''
            print 'these are the correct output values'
            print correct_output_values
            print ''

            original_results = [( yrow[0 : self.size_of_output] ) for yrow in y_test_data]

            # preprocess the dataset
            preprocessor    = construct_preprocessor( predict_dataset, [replace_nan, standarize] )
            predict_dataset = preprocessor( predict_dataset )

            # feed the instances to the network
            print 'these are the neural net predictions'
            results =  network.predict( predict_dataset ) # return a 2D NumPy array [n_samples, n_outputs]
            print [result[0 : self.size_of_output] for result in results]

            rounded_results = []
            for value_group in results:
                if self.size_of_output == 1:
                    rounded_results.append([round(value_group)])
                else:
                    grouping = []
                    for value in value_group:
                        grouping.append(round(value))
                    rounded_results.append(grouping)
            print ''
            print 'these are the rounded predictions'
            print rounded_results
            print ''

            accuracy = []
            for (original_group, NNpred_group) in zip(original_results, rounded_results):
                if len(original_group) == 1:
                    if original_group[0] != NNpred_group[0]:
                        accuracy.append(0)
                    else:
                        accuracy.append(1)
                else:
                    for (original, NNpred) in zip(original_group, NNpred_group):
                        if original != NNpred:
                            accuracy.append(0)
                        else:
                            accuracy.append(1)


            perc_accuracy = (sum(accuracy) / float(len(accuracy))) * 100
            perc_accuracy = self.truncate(perc_accuracy, 1)

            print 'percentage accuracy: ' + str(perc_accuracy) + "%"

            return correct_output_values, results, rounded_results

def run_ANN(train_dataset_location, test_dataset_location, size_of_output, num_of_neurons, cost_function, neuron_function, activation_function, training_function, max_iterations, max_error, weight_step_min, weight_step_max, learn_max, learn_min, learning_rate, start_step, momentum_factor, hidden_layer_dropout, input_layer_dropout, save_trained_network, network_name, saved_network_location, use_saved_network, view_NN_training):
    NN = NeuralNetwork(train_dataset_location, test_dataset_location, size_of_output, num_of_neurons, cost_function, neuron_function, activation_function, training_function, max_iterations, max_error)
    NN.prep_data(NN.X, NN.y)
    return NN.train_and_predict(NN.X, NN.y, NN.X_test_data, NN.y_test_data, weight_step_min, weight_step_max, learn_max, learn_min, learning_rate, start_step, momentum_factor, hidden_layer_dropout, input_layer_dropout, save_trained_network, network_name, saved_network_location, use_saved_network, view_NN_training)
