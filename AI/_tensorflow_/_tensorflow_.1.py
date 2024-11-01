import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model
model = Sequential()

# Input layer with first convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer (hidden layer)
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the pooled feature map into a single 1D vector
model.add(Flatten()) 

# Fully connected layer (hidden layer)
model.add(Dense(128, activation='relu'))

# Adding dropout to prevent overfitting
model.add(Dropout(0.5))

# Output layer (for binary classification, change units for multi-class classification)
model.add(Dense(128, activation='sigmoid'))

# Summary of the model architecture
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])