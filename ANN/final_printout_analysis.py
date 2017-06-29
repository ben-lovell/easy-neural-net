import numpy as np

def error_calculations(original_results, rounded_results, decimal_rounding_for_prediction):
    accuracy_percent = []
    accuracy_deviation = []
    for (original_group, NNpred_group) in zip(original_results, rounded_results):
        if len(original_group) == 1:
            if original_group[0] < NNpred_group[0]:
                accuracy_percent.append(original_group[0] / NNpred_group[0])
                accuracy_deviation.append( NNpred_group[0] - original_group[0])
            elif NNpred_group[0] == 0 and original_group[0] == 0:
                accuracy_percent.append(1)
                accuracy_deviation.append(0)
            else:
                accuracy_percent.append( NNpred_group[0] / original_group[0])
                accuracy_deviation.append(original_group[0] - NNpred_group[0])
        else:
            for (original, NNpred) in zip(original_group, NNpred_group):
                if original_group[0] < NNpred_group[0]:
                    accuracy_percent.append(original_group[0] / NNpred_group[0])
                    accuracy_deviation.append( NNpred_group[0] - original_group[0])
                else:
                    accuracy_percent.append( NNpred_group[0] / original_group[0])
                    accuracy_deviation.append(original_group[0] - NNpred_group[0])

    perc_accuracy = round((sum(accuracy_percent) / float(len(accuracy_percent))), 3) * 100

    print ('percentage accuracy: ' + str(perc_accuracy) + "%")
    print ("Average deviation: " + str(round(np.average(accuracy_deviation), decimal_rounding_for_prediction)))

def round_results(results, size_of_output, decimal_rounding_for_prediction):
    rounded_results = []
    for value_group in results:
        if size_of_output == 1:
            rounded_results.append([round(value_group, decimal_rounding_for_prediction)])
        else:
            grouping = []
            for value in value_group:
                grouping.append(round(value, decimal_rounding_for_prediction))
            rounded_results.append(grouping)
    print ('')
    print ('these are the rounded predictions')
    print (rounded_results)
    print ('')
    return rounded_results
