import time
import math
import numpy as np
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# This code uses MiniROCKET to extract features from R10 and X10 oscillometry data 
# and train a ridge regression classifier to make predictions on subjects' pulmonary function.

# MiniROCKET is run a specified number of times and probability voting is utilized to maximize prediction accuracy for each unique subject.

# Subjects are assigned a value of 0 if they are healthy or a value of 1 if they have restricted airways.

print(f' BEGIN MiniROCKET '.center(96, '='))

print(f'Loading data'.ljust(75, '.'), end = '', flush = True)

# Load the time series data.
_path_ = "C:/School/PhD/Data/OHV-ILD_10Hz/"
_r10_ohv_ = f"{_path_}OHV-R10_All_Trials_LABELLED.xlsx"
_x10_ohv_ = f"{_path_}OHV-X10_All_Trials_LABELLED.xlsx"
_r10_ild_ = f"{_path_}ILD-R10_All_Trials_LABELLED_v2.xlsx"
_x10_ild_ = f"{_path_}ILD-X10_All_Trials_LABELLED_v2.xlsx"
df_r10_ohv = pd.read_excel(_r10_ohv_)
df_x10_ohv = pd.read_excel(_x10_ohv_)
df_r10_ild = pd.read_excel(_r10_ild_)
df_x10_ild = pd.read_excel(_x10_ild_)

# Define a function that takes input DataFrames of time series data 
# and outputs labelled arrays of time series data ready for input to MiniROCKET.
# This function takes the following inputs:
#   - df_r10_ohv [DataFrame of R10 data from OHV subjects.]
#   - df_x10_ohv [DataFrame of X10 data from OHV subjects.]
#   - df_r10_ild [DataFrame of R10 data from ILD subjects.]
#   - df_x10_ild [DataFrame of X10 data from ILD subjects.]
# This function returns the following:
#   - r10_normal [NumPy array containing the subject ID and R10 time series data for healthy subjects.]
#   - x10_normal [NumPy array containing the subject ID and X10 time series data for healthy subjects.]
#   - r10_restriction [NumPy array containing the subject ID and R10 time series data for restriction subjects.]
#   - x10_restriction [NumPy array containing the subject ID and X10 time series data for restriction subjects.]
def get_labelled_arrays(df_r10_ohv, df_x10_ohv, df_r10_ild, df_x10_ild):
    master_r10 = np.array(df_r10_ohv)
    master_r10 = np.vstack((master_r10, np.array(df_r10_ild)))
    master_x10 = np.array(df_x10_ohv)
    master_x10 = np.vstack((master_x10, np.array(df_x10_ild)))
    
    r10_normal = np.empty((0, master_r10.shape[1]))
    x10_normal = np.empty((0, master_x10.shape[1]))
    r10_restriction = np.empty((0, master_r10.shape[1]))
    x10_restriction = np.empty((0, master_x10.shape[1]))
    for i in range(master_r10.shape[0]):
        subject_r10 = master_r10[i, :]
        if subject_r10[1] == 'NORMAL':
            r10_normal = np.vstack((r10_normal, subject_r10))
            x10_normal = np.vstack((x10_normal, master_x10[i, :]))
        elif subject_r10[1] == 'RESTRICTION':
            r10_restriction = np.vstack((r10_restriction, subject_r10))
            x10_restriction = np.vstack((x10_restriction, master_x10[i, :]))

    r10_normal = np.delete(r10_normal, slice(1, 7), 1)
    x10_normal = np.delete(x10_normal, slice(1, 7), 1)
    r10_restriction = np.delete(r10_restriction, slice(1, 7), 1)
    x10_restriction = np.delete(x10_restriction, slice(1, 7), 1)

    return r10_normal, x10_normal, r10_restriction, x10_restriction

# Define a function that takes input oscillometry time series data and returns 
# an array with elements containing each subject's ID and number of trials.
# The function takes the following input:
#   - ts_data [Master dataset - 2D array containing each subject's ID and trial number (element 0), and time series data.]
# The function returns the following:
#   - id_trials [A list of lists where each sublist contains a given subject's ID and number of trials.]
def subject_id_and_trials(ts_data):
    idx = 0
    id_trials = []
    while idx <= ts_data.shape[0]:
        cnt = 0
        current_trials = [] # List of trials for the current subject.
        current_sub = ts_data[idx] # Current subject ID and time series data.
        while current_sub[0][:7] == ts_data[idx+cnt][0][:7]:
            trial_no = cnt + 1 # Specific trial number of the current subject.
            current_trials.append(trial_no)
            cnt += 1
            if idx+cnt >= ts_data.shape[0]:
                break
        # Append the subject ID and the number of trials.
        id_trials.append([current_sub[0][:7], max(current_trials)])
        idx += cnt
        if idx >= ts_data.shape[0]:
            break

    return id_trials

# Define a function that takes input oscillometry time series data and returns shuffled "boosted" train and test sets.
# "Boosted" means that each time series has been partitioned into segments such that each segment can be considered a sub-trial of the original time series.
# The function takes the following inputs:
#   - r10_normal [2D array containing each healthy subject's ID and trial number (element 0), and their R10 time series data.] 
#   - x10_normal [2D array containing each healthy subject's ID and trial number (element 0), and their X10 time series data.]
#   - r10_restriction [2D array containing each restriction subject's ID and trial number (element 0), and their R10 time series data.]
#   - x10_restriction [2D array containing each restriction subject's ID and trial number (element 0), and their X10 time series data.]
#   - subjects_normal [int - Total number of (unique) healthy subjects.]
#   - subjects_restriction [int - Total number of (unique) restriction subjects.]
#   - train_ratio [float (0, 1) - Train / test split ratio.]
#   - divisor [int - Factor by which the original time series will be segmented.]  
# The function returns the following:
#   - X_train [3D array containing each training subject's ID and "boosted" data.]
#   - X_test [3D array containing each test subject's ID and "boosted" data.]
def create_X_train_and_test(r10_normal, x10_normal, r10_restriction, x10_restriction, subjects_normal, subjects_restriction, train_ratio, divisor):
    # Initialize all variables to be incremented. 
    i = 0
    j = 0
    k = 0
    l = 0
    n = 0
    id_and_ts_len = r10_normal.shape[1] # Length of the 1D array containing the subject's ID and time series data.
    train_subjects_normal = math.floor(train_ratio*subjects_normal) # Specified number of healthy subjects for the train set.
    train_subjects_restriction = math.floor(train_ratio*subjects_restriction) # Specified number of restrction subjects for the train set.
    train_subjects_total = train_subjects_normal + train_subjects_restriction # Total subjects in the train set.
    r10_train = np.empty((0, id_and_ts_len)) # Array containing the subject ID and R10 data for the train set.
    x10_train = np.empty((0, id_and_ts_len)) # Array containing the subject ID and X10 data for the train set.
    r10_test = np.empty((0, id_and_ts_len)) # Array containing the subject ID and R10 data for the test set.
    x10_test = np.empty((0, id_and_ts_len)) # Array containing the subject ID and X10 data for the test set.
    r10_master = np.vstack((r10_normal, r10_restriction)) # Master array of all subject ID's and R10 data.
    x10_master = np.vstack((x10_normal, x10_restriction)) # Master array of all subject ID's and X10 data.
    normal_restriction_ratio = subjects_normal / (subjects_normal + subjects_restriction) # Ratio of normal subjects to total subjects.
    ts_len_trunc = int((id_and_ts_len-1)/divisor) # Length of the truncated time series.
    r10_train_boosted = np.empty((0, 1+ts_len_trunc)) # Initialize an array to store the "boosted" R10 training data.
    x10_train_boosted = np.empty((0, 1+ts_len_trunc)) # Initialize an array to store the "boosted" X10 training data.
    r10_test_boosted = np.empty((0, 1+ts_len_trunc)) # Initialize an array to store the "boosted" R10 test data.
    x10_test_boosted = np.empty((0, 1+ts_len_trunc)) # Initialize an array to store the "boosted" X10 test data.
    #normal_restriction_ratio = r10_normal.shape[0] / (r10_normal.shape[0] + r10_restriction.shape[0])

    # Randomly select normal and restriction subjects and append all of their trials to 
    # r10_train and x10_train until they each contain the specified proportion of healthy and restriction subjects. 
    while i < train_subjects_total:
        np.random.seed(n)
        rnd_int = np.random.randint(1, 101) # Randomly select a number from 1-100.
        n += 1
        # If the randomly selected number is between 1 and the normal_restriction_ratio*100, 
        # select a random healthy subject and append their ID and R10 and X10 data to r10_train and x10_train. 
        if (rnd_int > 0 and rnd_int <= math.ceil(normal_restriction_ratio*100)) and j < train_subjects_normal:
            np.random.seed(l)
            rnd_idx = np.random.randint(0, r10_normal.shape[0]) # Randomly select an index of r10_normal.
            current_subject_train = r10_normal[rnd_idx] # Get the current subject at the random index.
            l += 1
            # Check to see if the current subject is already in r10_train.
            # If so, skip them and go to the next iteration of the loop.
            if current_subject_train in r10_train:
                continue
            else:
                subject_id = current_subject_train[0][:7] # Get the current subject's ID.
                # Append the data for all trials of the current subject from the master dataset of healthy subjects to the train set. 
                for jj in range(r10_normal.shape[0]):
                    master_id_normal = r10_normal[jj, 0][:7] # Subject's ID from the master dataset of healthy subjects.
                    if subject_id == master_id_normal:
                        r10_train = np.vstack((r10_train, r10_normal[jj]))
                        x10_train = np.vstack((x10_train, x10_normal[jj]))
                i += 1
                j += 1
        # If the randomly selected number is greater than the normal_restriction_ratio*100 and less than 100, 
        # select a random restriction subject and append their ID and R10 and X10 data to r10_train and x10_train. 
        elif (rnd_int > math.ceil(normal_restriction_ratio*100) and rnd_int <= 100) and k < train_subjects_restriction:
            np.random.seed(l)
            rnd_idx = np.random.randint(0, r10_restriction.shape[0]) # Randomly select an index of r10_restriction.
            current_subject_train = r10_restriction[rnd_idx] # Get the current subject at the random index.
            l += 1
            # Check to see if the current subject is already in r10_train.
            # If so, skip them and go to the next iteration of the loop.
            if current_subject_train in r10_train:
                continue
            else:
                subject_id = current_subject_train[0][:7] # Get the current subject's ID.
                # Append the data for all trials of the current subject from the master dataset of restriction subjects to the train set. 
                for kk in range(r10_restriction.shape[0]):
                    master_id_restriction = r10_restriction[kk, 0][:7] # Subject's ID from the master dataset of restriction subjects.
                    if subject_id == master_id_restriction:
                        r10_train = np.vstack((r10_train, r10_restriction[kk]))
                        x10_train = np.vstack((x10_train, x10_restriction[kk]))
                i += 1
                k += 1

    # Append the subject ID and time series data for all subjects not in r10_train and x10_train to r10_test and x10_test.
    for m in range(r10_master.shape[0]):
        current_subject_test = r10_master[m]
        if current_subject_test in r10_train:
            pass
        else:
            r10_test = np.vstack((r10_test, current_subject_test))
            x10_test = np.vstack((x10_test, x10_master[m]))

    # Iterate over each time series in r10_train and x10_train, partition the data into segments of equal length, 
    # and append the new time series to r10_train_boosted and x10_train_boosted.
    for p in range(r10_train.shape[0]):
        s_id_train = r10_train[p, 0]
        for q in range(divisor):
            temp_r10_train = np.array([])
            temp_x10_train = np.array([])
            temp_r10_train = np.append(temp_r10_train, s_id_train)
            temp_x10_train = np.append(temp_x10_train, s_id_train)
            temp_r10_train = np.append(temp_r10_train, r10_train[p, 1+q*ts_len_trunc:1+(q+1)*ts_len_trunc])
            temp_x10_train = np.append(temp_x10_train, x10_train[p, 1+q*ts_len_trunc:1+(q+1)*ts_len_trunc])
            r10_train_boosted = np.vstack((r10_train_boosted, temp_r10_train))
            x10_train_boosted = np.vstack((x10_train_boosted, temp_x10_train))

    # Iterate over each time series in r10_test and x10_test, partition the data into segments of equal length, 
    # and append the new time series to r10_test_boosted and x10_test_boosted.
    for r in range(r10_test.shape[0]):
        s_id_test = r10_test[r, 0]
        for s in range(divisor):
            temp_r10_test = np.array([])
            temp_x10_test = np.array([])
            temp_r10_test = np.append(temp_r10_test, s_id_test)
            temp_x10_test = np.append(temp_x10_test, s_id_test)
            temp_r10_test = np.append(temp_r10_test, r10_test[r, 1+s*ts_len_trunc:1+(s+1)*ts_len_trunc])
            temp_x10_test = np.append(temp_x10_test, x10_test[r, 1+s*ts_len_trunc:1+(s+1)*ts_len_trunc])
            r10_test_boosted = np.vstack((r10_test_boosted, temp_r10_test))
            x10_test_boosted = np.vstack((x10_test_boosted, temp_x10_test))

    # Assemble the train and test sets for input to MiniROCKET.
    # The time series data must be a 3D NumPy array with the following dimensions:
    # (sample_size, number_of_channels, time_series_length).
    X_train = np.array([r10_train_boosted, x10_train_boosted])
    X_train = np.transpose(X_train, (1, 0, 2))
    X_test = np.array([r10_test_boosted, x10_test_boosted])
    X_test = np.transpose(X_test, (1, 0, 2))

    return X_train, X_test

# Define a function that returns the labels for the train and test sets: y_train and y_test.
# The function takes the following inputs:
#   - X_train [3D array containing each training subject's ID and data.]
#   - X_test [3D array containing each test subject's ID and data.]
#   - r10_normal [2D array containing each healthy subject's ID and trial number (element 0), and their R10 time series data.] 
#   - r10_restriction [2D array containing each restriction subject's ID and trial number (element 0), and their R10 time series data.]
# The function returns the following:
#   - y_train [1D array containing the labels for all subject trials present in X_train.]
#   - y_test [1D array containing the labels for each unique subject present in X_test.]
def create_y_train_and_test(X_train, X_test, r10_normal, r10_restriction):
    y_train = np.array([])
    y_test = np.array([])

    # Loop over all subject trials in X_train and assign labels based on whether the subject is healthy or has restriction.
    for i in range(X_train.shape[0]):
        train_id = X_train[i, 0, 0]
        if train_id in r10_normal:
            y_train = np.append(y_train, 0)
        elif train_id in r10_restriction:
            y_train = np.append(y_train, 1)
    
    # Loop over all subjects in X_test and assign labels based on whether the subject is healthy or has restriction.
    # Labels are only assigned to unique subject ID's (i.e., all trials are considered holistically.)
    idx = 0
    while idx < X_test.shape[0]:
        cnt = 0
        test_id = X_test[idx, 0, 0]
        while test_id[:7] == X_test[idx+cnt, 0, 0][:7]:
            if cnt != 0:
                pass
            else:
                if test_id in r10_normal:
                    y_test = np.append(y_test, 0)
                elif test_id in r10_restriction:
                    y_test = np.append(y_test, 1)
            cnt += 1
            if idx+cnt >= X_test.shape[0]:
                break
        idx += cnt
    
    return y_train, y_test

# Define function performs probability voting on the predictions made by the ridge regression classifier.
# The function takes the following inputs:
#   - ensemble [1D NumPy array of classifier scores.]
#   - X_test [3D NumPy array containing each test subject's ID and data.]
# The function returns the following:
#   - voted_predictions [1D NumPy array of predictions that have been voted on for each unique subject.]
#   - class_probs [List of lists, where each sub-list contains the subject's ID [0], NORMAL probability [1], and RESTRICTION probability [2].]
def probability_voting(ensemble, X_test):
    idx = 0
    voted_predictions = np.array([]) # Array to store the predictions that have been voted on.
    class_probs = [] # List to store each subject's ID and classification probabilities ([0] --> ID, [1] --> Pr{NORMAL}, [2] --> Pr{RESTRICTION}).

    # Loop over the ensemble of classifier scores and vote on the predicted values for each subject.
    # Voting is implemented by selecting the class which attained the highest probability.
    while idx < ensemble.shape[0]:
        cnt = 0
        score_array = np.array([])
        subject_id = X_test[idx, 0, 0][:7]
        while subject_id == X_test[idx+cnt, 0, 0][:7]:
            scores = ensemble[idx+cnt]
            score_array = np.append(score_array, scores)
            cnt += 1
            if idx+cnt >= ensemble.shape[0]:
                break
        probs = np.empty((0, 2))
        for i in range(len(score_array)):
            if score_array[i] < 0:
                score_prob = np.exp(np.absolute(score_array[i]))/np.sum(np.exp(np.absolute(score_array)))
                probs = np.vstack((probs, np.array([score_prob, 0])))
            elif score_array[i] > 0:
                score_prob = np.exp(score_array[i])/np.sum(np.exp(np.absolute(score_array)))
                probs = np.vstack((probs, np.array([0, score_prob])))
        total_prob = np.array([np.sum(probs[:, 0]), np.sum(probs[:, 1])])
        class_probs.append([subject_id, total_prob[0], total_prob[1]])
        voted_predictions = np.append(voted_predictions, np.argmax(total_prob))
        
        idx += cnt
    
    return voted_predictions, class_probs

# Define a function that determines which subjects were misclassified.
# The function takes the following inputs:
#   - y_true [1D NumPy array of *correct* subject labels.]
#   - y_pred [1D NumPy array of *predicted* subject labels.]
#   - subject_ids [A list of lists where each sublist contains a given subject's ID and number of trials.]
# The function returns the following:
#   - misclassified_subjects [List misclassified subject ID's.]
def who_got_misclassified(y_true, y_pred, subject_ids):
    misclassified_subjects = []
    for i in range(len(subject_ids)):
        if y_true[i] != y_pred[i]:
            misclassified_subjects.append(subject_ids[i][0])

    return misclassified_subjects

# Get arrays of R10 and X10 time series data for healthy and restriction subjects.
r10_normal, x10_normal, r10_restriction, x10_restriction = get_labelled_arrays(df_r10_ohv, df_x10_ohv, df_r10_ild, df_x10_ild)

normal_subjects = len(subject_id_and_trials(r10_normal)) # Number of healthy subjects.
restriction_subjects = len(subject_id_and_trials(r10_restriction)) # Number of restriction subjects.

num_runs = 1 # Specify the number of runs to be performed by ROCKET and MiniROCKET.
divisor = 4 # Factor by which to divide the time series data.
ts_len = int((r10_normal.shape[1] - 1) / divisor) # New time series length (i.e., 280 - Needs to be divisible by multiples of 2).
split = 0.7 # Train / test split = 70 / 30.

# Array to store an ensemble of ridge estimators.
# Each row is are the residuals of the dot product of the subject's features (extracted from MiniROCKET) minus the bias for a given trial.
residual_ensemble = np.empty((0, 9996))

# Create empty arrays to store accuracy values.
accuracy_voting = np.array([])

# Create empty arrays to store elements of the confusion matrix for each run of MiniROCKET ("cm_ij").
cm_00 = np.array([])
cm_01 = np.array([])
cm_10 = np.array([])
cm_11 = np.array([])

scaler = StandardScaler()

print('Data has been loaded.')

#########################################################################################################################################
#########################################################################################################################################

# **********************************************************
# ******************** BEGIN MiniROCKET ********************
# **********************************************************

# Start the timer for ROCKET.
minirocket_start = time.time()

# Initialize MiniROCKET.
minirocket = MiniRocketMultivariate(num_features = 10_000, max_dilations_per_kernel = 32)

for i in range(num_runs):
    print(f'Performing run {i+1}'.ljust(80, '.'), end = '', flush = True)

    # Create shuffled train and test sets.
    X_train, X_test = create_X_train_and_test(r10_normal, x10_normal, r10_restriction, x10_restriction, normal_subjects, restriction_subjects, split, divisor)
    y_train, y_test = create_y_train_and_test(X_train, X_test, r10_normal, r10_restriction)

    X_train_data = X_train[:, :, 1:]
    X_test_data = X_test[:, :, 1:]

    #X_train_r10 = X_train[:, 0, 1:]
    #X_train_x10 = X_train[:, 1, 1:]
    #scaler.fit(X_train_r10)
    #X_train_r10_transform = scaler.transform(X_train_r10)
    #scaler.fit(X_train_x10)
    #X_train_x10_transform = scaler.transform(X_train_x10)
    #X_train_data = np.array([X_train_r10_transform, X_train_x10_transform])
    #X_train_data = np.transpose(X_train_data, (1, 0, 2))

    #X_test_r10 = X_test[:, 0, 1:]
    #X_test_x10 = X_test[:, 1, 1:]
    #scaler.fit(X_test_r10)
    #X_test_r10_transform = scaler.transform(X_test_r10)
    #scaler.fit(X_test_x10)
    #X_test_x10_transform = scaler.transform(X_test_x10)
    #X_test_data = np.array([X_test_r10_transform, X_test_x10_transform])
    #X_test_data = np.transpose(X_test_data, (1, 0, 2))

    # Create an empty array to store an ensemble of classifier scores.
    #scores_ensemble = np.empty((0, X_test.shape[0]))
    scores_ensemble = np.array([]) 
    
    # Transform the training data.
    minirocket.fit(X_train_data)
    X_train_transform = minirocket.transform(X_train_data)
        
    # Fit the classifier.
    classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    classifier.fit(X_train_transform, y_train)

    coefs = classifier.coef_ # Ridge coefficients.
    bias = classifier.intercept_[0] # Bias term.

    # Transform the test data.
    X_test_transform = minirocket.transform(X_test_data)

    # Iterate over the extracted features (from MiniROCKET) for each subject in X_test and calculate the individual residuals of each trace 
    # (i.e., the constituents of the inner product used to compute the classifier score).
    #for k in range(X_test_transform.shape[0]):
    #    res_array = np.array([])
    #    current_subject = np.array(X_test_transform)[k, :]
    #    for n in range(X_test_transform.shape[1]):
    #        res = current_subject[n]*coefs[0, n]
    #        res_array = np.append(res_array, res)
    #    residual_ensemble = np.vstack((residual_ensemble, res_array)) # Append the residuals to residual_ensemble.
        
    # Append predictions to the ensemble array.
    #scores_ensemble = np.vstack((scores_ensemble, classifier.decision_function(X_test_transform)))
    scores_ensemble = np.append(scores_ensemble, classifier.decision_function(X_test_transform))

    # Get the predicted values that have been voted on (voted_predictions) and the classification probabilities (class_probs).
    voted_predictions, class_probs = probability_voting(scores_ensemble, X_test)
    # Get the accuracy for the current run with voting.
    current_accuracy = accuracy_score(y_test, voted_predictions) 
    # Append MiniROCKET accuracy for the current run to the "minirocket_accuracy_voting" array.
    accuracy_voting = np.append(accuracy_voting, current_accuracy)

    # MiniROCKET confusion matrix.
    confusionMatrix = confusion_matrix(y_test, voted_predictions)

    # Store elements of the confusion matrix in their own NumPy array for each run.
    cm_00 = np.append(cm_00, confusionMatrix[0, 0])
    cm_01 = np.append(cm_01, confusionMatrix[0, 1])
    cm_10 = np.append(cm_10, confusionMatrix[1, 0])
    cm_11 = np.append(cm_11, confusionMatrix[1, 1])

    print(f'Run {i+1} complete.')

    print(confusionMatrix)
    
    print("Misclassified subjects:")
    
    misclassified_subjects = who_got_misclassified(y_test, voted_predictions, subject_id_and_trials(X_test[:, 0, :]))
    for subject in misclassified_subjects:
        print(subject)

# End the timer for MiniROCKET.
minirocket_finish = time.time()

# ********************************************************
# ******************** END MiniROCKET ********************
# ********************************************************

#########################################################################################################################################
#########################################################################################################################################

print(f' MiniROCKET COMPLETE '.center(96, '='))

# Average the accuracy of MiniROCKET over the specified number of trials.
accuracy_voting = np.average(accuracy_voting)

# MiniROCKET execution time.
minirocket_et = minirocket_finish - minirocket_start

# MiniROCKET confusion matrix (final, averaged).
MiniROCKET_Confusion = np.array([[np.average(cm_00), np.average(cm_01)],
                                 [np.average(cm_10), np.average(cm_11)]])

# Print the accuracy and execution time for MiniROCKET.

print('\n')

print(f' RESULTS '.center(96, '='))

print('''Average accuracy over {} runs with voting: {:.2%}.
Execution time: {:.4} s.
Confusion matrix:
{}'''.format(num_runs, accuracy_voting, minirocket_et, MiniROCKET_Confusion))

print(f''.center(96, '='))

#########################################################################################################################################
#########################################################################################################################################

# ******************************************************************************************
# ******************** PLOT HISTOGRAM OF TRACES FOR A SPECIFIED SUBJECT ********************
# ******************************************************************************************

#idx = 1 # Specify the index of a subject in y_test.
#test_id_and_trials = subject_id_and_trials(X_test[:, 0, :]) # Get the subject ID and number of trials for each subject in X_test.

#subject = test_id_and_trials[idx][0] # Subject ID at idx.
#trials = test_id_and_trials[idx][1] # Subject trials at idx.

#print(f'Subject ID: {subject}')
#print(f'Trials = {trials}')

## Find the number of cumulative trials (i.e., cum_trials) up to the specified test subject.
#cum_trials = 0
#for i in range(len(test_id_and_trials)):
#    cur_sub = test_id_and_trials[i]
#    if cur_sub[0] == test_id_and_trials[idx][0]:
#        break
#    else:
#        cum_trials += cur_sub[1]

## Get the residuals for each of the specified subject's traces.
#trial_cnt = 0
#subject_residuals = np.empty((0, residual_ensemble.shape[1]))
#while trial_cnt < trials:
#    current_residuals = residual_ensemble[idx+cum_trials+trial_cnt, :]
#    subject_residuals = np.vstack((subject_residuals, current_residuals))
#    trial_cnt += 1

## Plot the histogram of each trace with the mean (\mu), variance (\sigma^2), and classifier score.
#trial_idx = 0
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = 'century'
#plt.rcParams['mathtext.fontset'] = 'cm'
#while trial_idx < trials:
#    plt.figure()
#    plt.hist(subject_residuals[trial_idx], bins = 5, color = 'lightsteelblue', edgecolor = 'navy')
#    plt.title('Histogram of Trace {} for Subject {}\n$\mu = {:.4}$, $\sigma^2 = {:.4}$, Score $= {:.4}$'. format(trial_idx+1, 
#                                                                                                                 subject,
#                                                                                                                 np.average(subject_residuals[trial_idx, :]), 
#                                                                                                                 np.var(subject_residuals[trial_idx, :]),
#                                                                                                                 np.sum(subject_residuals[trial_idx, :]) + bias))
#    plt.xlabel(r'$w_{i} \cdot \mathrm{PPV}_{i} - b$')
#    plt.ylabel('Counts')

#    trial_idx += 1

#plt.show()

#########################################################################################################################################
#########################################################################################################################################

# ***********************************************************************************************
# ******************** PLOT CLASSIFIER SCORES FOR EACH MISCLASSIFIED SUBJECT ********************
# ***********************************************************************************************

#print('\n')

#print("Subject ID's [0] and classification probabilities ([1] --> 'NORMAL', [2] --> 'RESTRICTION'):")
#for i in range(len(class_probs)):
#    print(class_probs[i])

#print('\n')

#test_id_and_trials = subject_id_and_trials(X_test[:, 0, :]) # Get the subject ID and number of trials for each subject in X_test.

## Plot the classifier scores for each misclassified subject.
#for subject in misclassified_subjects:
#    idx = 0 # Initialize the misclassified subject index.
#    # Get the index of the misclassified subject in y_test --> corresponds to their index in test_id_and_trials.
#    for i in range(len(test_id_and_trials)):
#        if subject == test_id_and_trials[i][0]:
#            break
#        idx += 1 # Increment the index.
    
#    test_id = test_id_and_trials[idx][0] # Get the misclassified subject's ID from the list of test subjects.
#    test_trials = test_id_and_trials[idx][1] # Get the misclassified subject's number of trials from the list of test subjects.

#    cnt = 0 # Initialize the trial (index) count.
#    # Get the index of the misclassified subject's classifier score in scores_ensemble via counting the total number of trials up to their index.
#    for sub in test_id_and_trials:
#        id = sub[0]
#        trials = sub[1]
#        if id == test_id:
#            break
#        cnt += trials

#    subject_traces = scores_ensemble[cnt:cnt+test_trials] # Initialize an array to store the misclassified subject's classifier scores.

#    # Get the misclassified subject's classifier scores for each of their trials.
#    #for j in range(divisor):
#    #    start = cnt
#    #    end = cnt + test_trials
#    #    subject_traces = np.append(subject_traces, scores_ensemble[j, start:end])

#    cnt_pos = 0 # Initialize the misclassified subject's number of positive classifier scores. 
#    cnt_neg = 0 # Initialize the misclassified subject's number of negative classifier scores.
#    for trace in subject_traces:
#        if trace > 0:
#            cnt_pos += 1
#        elif trace < 0:
#            cnt_neg +=1
    
#    n_traces = [x+1 for x in range(len(subject_traces))] # Get the total number of traces.

#    area = np.trapz(subject_traces, n_traces, dx = 0.05) # Calculate the area under the ridge estimator curve (AUC).

#    plt.rcParams['font.family'] = 'serif'
#    plt.rcParams['font.serif'] = 'century'
#    plt.rcParams['mathtext.fontset'] = 'cm'

#    # Plot the classifier scores as a function of trace index.
#    plt.figure()
#    plt.plot([], [], 'k_', label = r'$\hat{\beta_{\:}} \: < \: 0 \: \rightarrow \: \mathrm{NORMAL}$')
#    plt.plot([], [], 'k+', label = r'$\hat{\beta_{\:}} \: > \: 0 \: \rightarrow \: \mathrm{RESTRICTION}$')
#    plt.scatter(n_traces, subject_traces, s = 20, color = 'navy', zorder = 4)
#    #plt.scatter(n_traces, subject_traces, s = 50, color = 'white', linewidths = 0.5, edgecolors = 'navy', label = 'Ridge Estimator / Trace Index')
#    plt.axhline(y = 0, color = 'red', linewidth = 1.5, label = 'Decision Boundary')
#    plt.plot(n_traces, subject_traces, color = 'navy', linewidth = 1.0, label = r'Ridge Estimator $i$, $\hat{\beta_{i}}$')
#    plt.fill_between(n_traces, subject_traces, color = 'lightsteelblue')
#    plt.xlabel(r'Trace Index, $i$', fontsize = 11)
#    plt.ylabel(r'Ridge Estimator, $\hat{\beta_{\:}}$', fontsize = 11)
#    plt.title('Ridge Estimator Values for Subject {}\nPositive Traces = {:}, Negative Traces = {:}'.format(test_id, cnt_pos, cnt_neg))
#    plt.legend(loc = 'best', fontsize = 9, labelspacing = 0.25).get_frame().set_edgecolor('k')
#    plt.tight_layout()

#    print(class_probs[idx][0] + ':', 'P(normal) = {:.2%},'.format(class_probs[idx][1]), 'P(restriction) = {:.2%},'.format(class_probs[idx][2]), 'AUC = {:.4}'.format(area))

#plt.show()