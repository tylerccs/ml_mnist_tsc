# this is the main file for creating the network and using it on the mnist dataset

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# Load the mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape the images
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Save the model
model.save('mnist_model.h5')

# export the model to a file
# Load the model
model = keras.models.load_model('mnist_model.h5')

# Make predictions
predictions = model.predict(x_test[:10])
print(predictions)


#create a summary document of the model with example images and predictions with some metrics
# Plot the images and predictions
for i in range(10):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Prediction: {np.argmax(predictions[i])}')
    plt.show()
    
# Calculate the accuracy
predictions = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test[:10], axis=1)
accuracy = np.mean(predictions == true_labels)
print(f'Accuracy: {accuracy}')

# Calculate the precision, recall, and F1 score
tp = np.sum((predictions == 1) & (true_labels == 1))
fp = np.sum((predictions == 1) & (true_labels == 0))
fn = np.sum((predictions == 0) & (true_labels == 1))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Save the metrics to a file
# create the file and write the metrics to it
with open('metrics.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy}\n')
    file.write(f'Precision: {precision}\n')
    file.write(f'Recall: {recall}\n')
    file.write(f'F1 Score: {f1}\n')
