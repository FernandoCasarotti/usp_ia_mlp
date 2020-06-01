# Imports
import keras
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Dictionary to facilitate the relation of each class to a number
letters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'J': 5, 'K': 6, '0': 0, '1': 1}

# Load the train data
X_train = [line.split(',') for line in open('src_problemXOR.csv')]
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

# Load the test data
X_test = [line.split(',') for line in open('src_problemXOR.csv')]
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

# Static configuration options
feature_vector_length = columns
num_classes = rows
possible_results = len(set(list(Y_train)))

# Custom configuration options; different handling depending if the problem is binary or categorical
# doesn't change the architecture, just adapt a bunch of parameters for better performance of the algorithm
# Obs.: the only architectural change is the size of the output layer, which is naturally different
if possible_results == 2:
    accuracy = 'binary_accuracy'
    validation_split = 0.00
    loss = 'binary_crossentropy'
    batch = 4
    output_neurons = 1
elif possible_results == 7:
    accuracy = 'accuracy'
    validation_split = 0.30
    loss = 'categorical_crossentropy'
    batch = 2
    output_neurons = 7
    # Convert target classes to categorical ones
    Y_train = to_categorical(Y_train, 7)
    Y_test = to_categorical(Y_test, 7)

# Set the input shape
input_shape = (feature_vector_length,)
print(f'Feature shape: {input_shape}')

# Create the model with one hidden layer connected to the input layer with the size of the number of columns
# the output layer connects to the hidden by default, with the size of the output length
# Obs.: this number of nodes on the hidden layer is the most balanced one with this optimizer, some variations
# resolves the characters in fewer epochs, but the XOR/OR/AND takes more on the other hand
model = Sequential()
model.add(Dense(units=14, use_bias=True, input_dim=feature_vector_length, activation='sigmoid'))
model.add(Dense(output_neurons, activation='sigmoid'))

# Configure the model and start training
# Obs.: the number of epochs needed can already be lower than it is at this time, the accuracy needed is reached
# far earlier, and that's because of the optimization. If a higher performance is needed, we could add one more hidden
# layer with 28 nodes on each, or just increase the single one with 56, changing the optimizer to 'adam'
model.compile(loss=loss, optimizer='adadelta', metrics=[accuracy])
model.fit(X_train, Y_train, epochs=1000, batch_size=batch, verbose=2, validation_split=validation_split)

# Custom testing of the model depending if the data is binary or categorical to produce the results
if possible_results == 7:
    test_results = model.evaluate(X_test, Y_test, verbose=1)
    print(f'Test results for characters - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
elif possible_results == 2:
    print(model.predict(X_train, verbose=1, batch_size=4))