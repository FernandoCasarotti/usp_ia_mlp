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

def nnaisolver(problemType):

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

    # Dictionary to facilitate the relation of each class to a number
    letters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'J': 5, 'K': 6, '0': 0, '1': 1}

    # Dictionary to facilitate the relation of each entrance to a file
    problemFiles = {'character': 'src_caracteres-limpo.csv', 'xor': 'src_problemXOR.csv', 'or': 'src_problemOR.csv', 'and': 'src_problemAND.csv'}

    # Load the train data
    X_train = [line.split(',') for line in open(problemFiles[problemType])]
    X_train[0][0] = X_train[0][0].replace('\ufeff', '')
    rows = len(X_train)
    columns = len(X_train[0])-1
    Y_train = []

    # Treat and format the train data
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

    # Custom configuration options; establish parameters to solve categorical character problem
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

    # Custom configuration options; establish parameters to solve the binary problems in better performance
    else:
        accuracy = 'binary_accuracy'
        validation_split = 0.00
        loss = 'binary_crossentropy'
        batch = 4
        output_neurons = 1

    # Static configuration options
    feature_vector_length = columns
    hidden_shape = 14
    epochs = 1000
    # Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)
    # Sigmoid is equivalent to a 2-element Softmax, where the second element is assumed to be zero.
    # The sigmoid function always returns a value between 0 and 1
    activator = 'sigmoid'
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
    # For learning to happen, we need to train our model with sample input/output pairs, such learning is called supervised learning
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, verbose=2, validation_split=validation_split)

    # After training the neural network, print the final weights of the already trained network
    fweights = model.get_weights()
    countf = 0
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
    if problemType == 'character':
        test_results = model.evaluate(X_test, Y_test, verbose=1)
        print(f'Test results for characters - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

    # Predict the results and generate the file for predictions
    # For Keras, the predictions are summarized in a relation between the existent samples and the output neurons,
    # which means that we have how were the predictions for the other results, not only the actual chosen one
    prediction = model.predict(X_train, verbose=1, batch_size=batch)
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

    # Just print the main details of the model
    model.summary()

if __name__ == '__main__':
    # A little argparse to create a carefully help description for the function
    parser = argparse.ArgumentParser(
        description='''
           For this function to run, it needs a unique argument with a string containing the type of problem to solve
           using the NN MLP algorithm built.
           Valid entrances are 'character', "xor", "or" and "and"''',
        epilog="""Future advanced features will be added later, including new arguments, will be included later""")
    parser.add_argument('problem', type=str, default="notInserted", help='problemToSolve')
    args = parser.parse_args()

    # Handling the possible arguments passed
    if sys.argv[1] == "-h" or sys.argv[1] == '--help':
        pass
    elif sys.argv[1] != 'character' and sys.argv[1] != 'xor' and sys.argv[1] != 'or' and sys.argv[1] != 'and':
       raise ValueError(
           "Insert a valid entrance: the types of problem that this algorithm solve are 'character', 'xor', 'or' and 'and'")
    else:
       problemType = sys.argv[1]
       nnaisolver(problemType)