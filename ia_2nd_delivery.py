""" Integrantes
 Fernando Gardim Casarotto
 Matheus Marin de Oliveira
 Isabelle Neves Porto
 Thiago de Oliveira Deodato
 Caique Novaes Pereira
"""

# System imports
import sys
import argparse

from matplotlib.backends.backend_pdf import PdfPages


def gradeanalysis():
    # Define general parameters for the grade-based analysis
    neurons = [5, 20, 40, 60, 80]
    lr = [1.0, 0.5, 0.3, 0.1]
    epochs = [50, 100, 150]
    count = 1

    # Loop to get an analysis through all the combinations possible
    for n in neurons:
        for l in lr:
            for e in epochs:
                title = "\nExecution n."+str(count)+" - HiddenNeurons: "+str(n)+"; Lr: "+str(l)+"; Epochs: "+str(e)+"; \n\n"
                params = {'neuron': n, 'lr': l, 'epoch': e, 'string': title}
                nnaisolver('votes_grade', params)
                count = count + 1

def nnaisolver(problemType, params):

    # Build  reproducibility
    #  Set a seed value
    seed_value = 28904
    #  1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    #  2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)
    #  3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)
    #  4. Set `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.compat.v1.set_random_seed(seed_value)
    #  5 Configure a new global `tensorflow` session
    from keras import backend as K
    #  If you have an operation that can be parallelized internally,
    #  TensorFlow will execute it by scheduling tasks in a thread pool with intra_op_parallelism_threads threads
    #  If you have many operations that are independent in your TensorFlow graph—because there is no directed
    #  path between them in the dataflow graph—TensorFlow will attempt to run them concurrently, using a thread pool
    #  If 0 is passed, the system picks an appropriate number, which means that each thread pool will have one thread per CPU core in your machine.
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

    # Imports for modelling the nn
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import to_categorical
    from keras.optimizers import Adadelta
    from keras.models import load_model
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from matplotlib import pyplot
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support
    from matplotlib.backends.backend_pdf import PdfPages

    # Dictionary to facilitate the relation of each class to a number
    letters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'J': 5, 'K': 6, '0': 0, '1': 1, 'republican': 0, 'democrat': 1, 'republican\n': 0, 'democrat\n': 1}

    # Dictionary to facilitate the relation of each entrance to a file
    problemFiles = {
        'character': 'src_caracteres-limpo.csv',
        'xor': 'src_problemXOR.csv',
        'or': 'src_problemOR.csv',
        'and': 'src_problemAND.csv',
        'votes': 'house-votes-84.csv',
        'votes_grade': 'house-votes-84.csv',
        'votes_chosen': 'house-votes-84.csv',
        'votes_chosen_es': 'house-votes-84.csv'
    }

    # Load the generals of the train data dynamically
    X_train = [line.split(',') for line in open(problemFiles[problemType])]
    rows = len(X_train)
    columns = len(X_train[0])-1
    Y_train = []

    # Treat specifically the data source of the votes problem, changing the class to numerical and each column value to a number
    if problemType == 'votes' or problemType == 'votes_grade' or problemType == 'votes_chosen' or problemType == 'votes_chosen_es':
        # Treat and format the train data with 60% of the whole available
        for x in range(0, 261):
            classRow = X_train[x][-1]
            X_train[x].pop()
            Y_train.append(letters[str(classRow)])
            for y in range(0, columns):
                if str(X_train[x][y]) == 'y':
                    X_train[x][y] = 1
                elif str(X_train[x][y]) == 'n':
                    X_train[x][y] = -1
                else:
                    X_train[x][y] = 0
        X_train = X_train[:261]
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
    # Treat the data for the other problems, changing the string value to a number and the string class as a number
    else:
        for x in range(0, rows):
            classRow = X_train[x][-1]
            classRow = classRow.replace('\n', '')
            X_train[x].pop()
            Y_train.append(letters[str(classRow)])
            for y in range(0, columns):
                try:
                    int(str(X_train[x][y]))
                    X_train[x][y] = int(str(X_train[x][y]))
                except ValueError:
                    X_train[x][y] = float(str(X_train[x][y]))
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

    # Custom configuration options; establish parameters to solve categorical character problem and the test data
    if problemType == 'character':
        # Load the test data
        X_test = [line.split(',') for line in open('src_caracteres-ruido.csv')]
        X_test[0][0] = X_test[0][0].replace('\ufeff', '')
        rowsT = len(X_test)
        columnsT = len(X_test[0])-1
        Y_test = []

        # Treat and format the test data
        for x in range(0, rowsT):
            classRowT = X_test[x][-1]
            classRowT = classRowT.replace('\n', '')
            X_test[x].pop()
            Y_test.append(letters[str(classRowT)])
            for y in range(0, columnsT):
                try:
                    int(str(X_test[x][y]))
                    X_test[x][y] = int(str(X_test[x][y]))
                except ValueError:
                    X_test[x][y] = float(str(X_test[x][y]))
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        accuracy = 'accuracy'
        validation_split = 0.30
        # Computes the crossentropy loss between the labels and predictions, with labels in one-hot
        loss = 'categorical_crossentropy'
        # batch size is important to apply stochastic gradient descent[sgd]
        batch = 2
        output_neurons = 7
        # Convert target classes to categorical ones
        Y_train = to_categorical(Y_train, 7)
        Y_test = to_categorical(Y_test, 7)

    # Custom configuration options; establish parameters to solve the binary vote problem and the test data
    # Obs.: this can get better increasing the batch size and trying harder on the validation_split
    elif problemType == 'votes' or problemType == 'votes_grade' or problemType == 'votes_chosen' or problemType == 'votes_chosen_es':
        # Load the test data
        X_test = [line.split(',') for line in open('house-votes-84.csv')]
        rowsT = len(X_test)
        columnsT = len(X_test[0]) - 1
        Y_test = []

        # Treat and format the test data
        for x in range(261, 435):
            classRowT = X_test[x][-1]
            X_test[x].pop()
            Y_test.append(letters[str(classRowT)])
            for y in range(0, columnsT):
                if str(X_test[x][y]) == 'y':
                    X_test[x][y] = 1
                elif str(X_test[x][y]) == 'n':
                    X_test[x][y] = -1
                else:
                    X_test[x][y] = 0
        X_test = X_test[-174:]
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        accuracy = 'binary_accuracy'
        validation_split = 0.30
        loss = 'binary_crossentropy'
        # batch size is important to apply stochastic gradient descent[sgd]
        batch = 2
        output_neurons = 1

    # Custom configuration options; establish parameters to solve the binary problems in better performance
    else:
        accuracy = 'binary_accuracy'
        validation_split = 0.00
        loss = 'binary_crossentropy'
        batch = 4
        output_neurons = 1

    # Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)
    # Sigmoid is equivalent to a 2-element Softmax, where the second element is assumed to be zero.
    # The sigmoid function always returns a value between 0 and 1
    activator = 'sigmoid'
    # Static configuration option; personalize it in case of grade analysis, but if not, apply the default
    feature_vector_length = columns
    if problemType == 'votes_grade' or problemType == 'votes_chosen' or problemType == 'votes_chosen_es':
        hidden_shape = params['neuron']
        epochs = params['epoch']
        optimizer = Adadelta(lr=params['lr'])
    else:
        hidden_shape = 14
        epochs = 1000
        # Adadelta optimization is a stochastic gradient descent method that is based on
        # adaptive learning rate per dimension
        # Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving
        # window of gradient updates, instead of accumulating all past gradients. This way, Adadelta
        # continues learning even when many updates have been done.
        optimizer = 'adadelta'

    # Create the model with one hidden layer connected to the input layer with the size of the number of columns
    # the output layer connects to the hidden by default, with the size of the output length
    # Obs.: this number of nodes on the hidden layer is the most balanced one with this optimizer, some variations
    # resolves the characters in fewer epochs, but the XOR/OR/AND takes more on the other hand
    model = Sequential()
    # Dense implements the operation: output = activation(dot(input, kernel) + bias)
    model.add(Dense(units=hidden_shape, use_bias=True, input_dim=feature_vector_length, activation=activator))
    model.add(Dense(output_neurons, activation=activator))

    # Configure the model and start training
    # Obs.: the number of epochs needed can already be lower than it is at this time, the accuracy needed is reached
    # far earlier, and that's because of the optimization. If a higher performance is needed, we could add one more
    # hidden layer with 28 nodes on each, or just increase the single one with 56, changing the optimizer to 'adam'
    model.compile(loss=loss, optimizer=optimizer, metrics=[accuracy])
    nnLr = K.eval(model.optimizer.lr)

    # Printing neural network main properties
    if problemType == 'votes_grade':
        nnParameters = open("nn_parameters.txt", "a")
        nnParameters.write(params['string'])
    else:
        nnParameters = open("nn_parameters.txt", "w+")
    nnParameters.writelines(["Number of entrance neurons: %s\n" % feature_vector_length,
                             "Number of hidden neurons: %s\n" % hidden_shape,
                             "Number of exit neurons: %s\n" % output_neurons,
                             "Number of epochs: %s\n" % epochs,
                             "Initial learning rate: %s\n" % nnLr,
                             "Activation function: %s\n" % activator,
                             "Optimizer: %s\n" % optimizer,
                             "Cost function: %s\n" % loss,
                             "Random seed: %s\n" % seed_value])
    nnParameters.close()


    # Before training the neural-network, print the initial weights of the already built network
    iweights = model.get_weights()
    counti = 0
    if problemType == 'votes_grade':
        initialWeights = open("initial_Weights.txt", "a")
        initialWeights.write(params['string'])
    else:
        initialWeights = open("initial_Weights.txt", "w+")
    # Since the function of weights returns a list of 4 arrays
    # (2 for hidden layer weights and bias, 2 for output layer weights and bias), we loop them but treat then different
    for arr in iweights:
        if counti == 0:
            initialWeights.write("Hidden layer weights (ixj): \n")
            # The hidden layer is a 2d array, with one line per entrance column and one column per neuron
            for i in range(0, len(arr)):
                for j in range(0, len(arr[i])):
                    initialWeights.write("\t%i~%i: %s\n" % (i, j, str(arr[i][j])))
        elif counti == 1:
            initialWeights.write("\nHidden layer bias: \n")
            # The hidden layer bias has a bias per neuron
            for i in range(0, len(arr)):
                initialWeights.write("\t%i: %s\n" % (i, str(arr[i])))
        elif counti == 2:
            initialWeights.write("\nOutput layer weights (ixj): \n")
            for i in range(0, len(arr)):
                for j in range(0, len(arr[i])):
                    initialWeights.write("\t%i~%i: %s\n" % (i, j, str(arr[i][j])))
        elif counti == 3:
            initialWeights.write("\nOutput layer bias: \n")
            for i in range(0, len(arr)):
                initialWeights.write("\t%i: %s\n" % (i, str(arr[i])))
        counti = counti + 1
    initialWeights.close()

    # Execute the training of the neural network, printing the predictions made on each epoch and saving the history
    # For learning to happen, we need to train our model with sample input/output pairs,
    # such learning is called supervised learning

    # If the problem to solve involves early stopping, then stop based on mse but waits 5 epochs to understand if
    # it gets better; if not, load the best result
    if problemType == 'votes_chosen_es':
        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, verbose=2, validation_split=validation_split,
                            callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10),
                                       ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
        )
        model = load_model('best_model.h5')
    # General case of training
    else:
        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, verbose=2, validation_split=validation_split)

    # After training the neural network, print the final weights of the already trained network
    fweights = model.get_weights()
    countf = 0
    if problemType == 'votes_grade':
        finalWeights = open("final_Weights.txt", "a")
        finalWeights.write(params['string'])
    else:
        finalWeights = open("final_Weights.txt", "w+")
    for arr in fweights:
        if countf == 0:
            finalWeights.write("Hidden layer weights (ixj): \n")
            for i in range(0, len(arr)):
                for j in range(0, len(arr[i])):
                    finalWeights.write("\t%i~%i: %s\n" % (i, j, str(arr[i][j])))
        elif countf == 1:
            finalWeights.write("\nHidden layer bias: \n")
            for i in range(0, len(arr)):
                finalWeights.write("\t%i: %s\n" % (i, str(arr[i])))
        elif countf == 2:
            finalWeights.write("\nOutput layer weights (ixj): \n")
            for i in range(0, len(arr)):
                for j in range(0, len(arr[i])):
                    finalWeights.write("\t%i~%i: %s\n" % (i, j, str(arr[i][j])))
        elif countf == 3:
            finalWeights.write("\nOutput layer bias: \n")
            for i in range(0, len(arr)):
                finalWeights.write("\t%i: %s\n" % (i, str(arr[i])))
        countf = countf+1
    finalWeights.close()

    if problemType == 'votes_grade':
        errors = open("errors.txt", "a")
        errors.write(params['string'])
    else:
        errors = open("errors.txt", "w+")
    counte = 0
    # The history of errors logged by Keras summarizes the losses of all the neurons of the layers to produce
    # one loss value per epoch (i.e. cost)
    for line in history.history['loss']:
        errors.write("Epoch %i.\n" % counte)
        errors.write("\t Error: %s\n\n" % line)
        counte = counte+1
    errors.close()

    # Custom testing for categorical problems, feeding the trained model with the test data and calculating the accuracy
    # for the samples over the epochs
    if problemType == 'character' or problemType == 'votes':
        test_results = model.evaluate(X_test, Y_test, verbose=1)
        print(f'Test results for {problemType} - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
    elif problemType == 'votes_grade' or problemType == 'votes_chosen' or problemType == 'votes_chosen_es':
        test_results = model.evaluate(X_test, Y_test, verbose=1)
        print(f'Test results for {problemType} - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

    # Predict the results and generate the file for predictions
    # For Keras, the predictions are summarized in a relation between the existent samples and the output neurons,
    # which means that we have how were the predictions for the other results, not only the actual chosen one
    prediction = model.predict(X_train, verbose=1, batch_size=batch)
    if problemType == 'votes_grade':
        predictedValues = open("predictions.txt", "a")
        predictedValues.write(params['string'])
    else:
        predictedValues = open("predictions.txt", "w+")
    countpx = 0
    for x in prediction:
        predictedValues.write("\nSample %i.\n" % countpx)
        countpy = 0
        for y in x:
            predictedValues.write("\tOutput neuron index %i: %s\n" % (countpy, y))
            countpy = countpy + 1
        countpx = countpx + 1
    predictedValues.close()

    if problemType == 'votes_grade':
        # Plot the relation between loss by epoch in training and validation
        with PdfPages('lossByEpoch.pdf') as pdfL:
            pyplot.figure(1)
            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history['val_loss'], label='test')
            pyplot.xlabel('Loss')
            pyplot.ylabel('Epoch')
            pyplot.legend()
            titleString = 'Loss by Epoch (' + params['string'] + ')'
            pyplot.title(titleString)
            pdfL.savefig()
            pyplot.close()
    else:
        pyplot.figure(1)
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.xlabel('Loss')
        pyplot.ylabel('Epoch')
        pyplot.legend()
        pyplot.title('Loss by Epoch')
        pyplot.show()

    # Plot the ROC curve
    y_pred_keras = model.predict(X_test, verbose=1, batch_size=batch).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    if problemType == 'votes_grade':
        with PdfPages('roc.pdf') as pdf:
            pyplot.figure(1)
            pyplot.plot([0, 1], [0, 1], 'k--')
            pyplot.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
            pyplot.xlabel('False positive rate')
            pyplot.ylabel('True positive rate')
            pyplot.legend(loc='best')
            pyplot.title('ROC curve ('+params['string']+')')
            pdf.savefig()  # saves the current figure into a pdf page
            pyplot.close()
    else:
        pyplot.figure(1)
        pyplot.plot([0, 1], [0, 1], 'k--')
        pyplot.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        pyplot.xlabel('False positive rate')
        pyplot.ylabel('True positive rate')
        pyplot.legend(loc='best')
        pyplot.title('ROC curve')
        pyplot.show()

    # Print the confusion matrix
    val_predicts = model.predict(X_test)
    y_pred = [1 * (x[0]>=0.5) for x in val_predicts]
    cm = confusion_matrix(Y_test, y_pred)

    if problemType == 'votes_grade':
        precisionValues = open("precision.txt", "a")
        precisionValues.write(params['string'])
    else:
        precisionValues = open("precision.txt", "w+")

    # here we have simple check where we assume that all people are healthy
    # then, we check what probability we get with dumb guess

    # Normal cases can be counted by summing all labels that are zeros
    precisionValues.write('Simple guess accuracy was: {:.4f}\n'.format(np.sum(Y_test == 0) / len(Y_test)))

    # Accuracy can be calculated from the confusion matrix by
    # counting all elements in diagonal (=trace of the matrix)
    traceCm = np.trace(cm)
    sumCm = sum(cm)
    ttAcc = traceCm / sumCm
    precisionValues.write('Total accuracy was: %s\n' % str(ttAcc))

    # __________ | Not Predicted | Predicted
    # Not Actual |               |
    # Yes Actual |               |
    precisionValues.write('Confusion Matrix:\n')
    precisionValues.write(str(cm))

    # Print precision, recall, fscore and support
    p, r, f, s = precision_recall_fscore_support(Y_test, y_pred)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    precisionValues.write('\nSupport:\n')
    precisionValues.write(str(s))
    precisionValues.write('\nPrecision: ')
    precisionValues.write(str(p))
    precisionValues.write('\nRecall: ')
    precisionValues.write(str(r))
    precisionValues.write('\nF-score: ')
    precisionValues.write(str(f))
    precisionValues.write('\n')

    precisionValues.close()


if __name__ == '__main__':
    # A little argparse to create a carefully help description for the function
    parser = argparse.ArgumentParser(
        description='''
           For this function to run, it needs a unique argument with a string containing the type of problem to solve
           using the NN MLP algorithm built.
           Valid entrances are 'character', "xor", "or", "and", "votes", "votes_grade", "votes_chosen" and "votes_chosen_es" ''',
        epilog="""Future advanced features will be added later, including new arguments, will be included later""")
    parser.add_argument('problem', type=str, default="notInserted", help='problemToSolve')
    args = parser.parse_args()

    # Handling the possible arguments passed
    problemType = sys.argv[1]
    if problemType == "-h" or problemType == '--help':
        pass
    elif problemType == 'votes_grade':
        gradeanalysis()
    elif problemType == 'votes_chosen_es' or problemType == 'votes_chosen':
        params = {'neuron': 40, 'lr': 1.0, 'epoch': 50}
        nnaisolver(problemType, params)
    elif problemType != 'character' and problemType != 'xor' and problemType != 'or' and problemType != 'and' and problemType != 'votes' and problemType != 'votes_grade' and problemType != 'votes_chosen_es' and problemType != 'votes_chosen':
       raise ValueError(
           "Insert a valid entrance: the types of problem that this algorithm solve are 'character','xor', 'or', 'and', 'votes', 'votes_grade', 'votes_chosen' and 'votes_chosen_es'")
    else:
        params = {}
        nnaisolver(problemType, params)
