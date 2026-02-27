from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from Evaluation import evaluation
def build_dtwn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def Model_DTWN(Train_Data, Train_Target, Test_Data, Test_Target):
    input_shape = (100, 1)  # Input shape, assuming time series data with 100 timesteps and 1 feature
    num_classes = 10  # Number of classes
    model = build_dtwn(input_shape, num_classes)
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model (assuming you have your training data X_train and labels y_train)
    model.fit(Train_Data, Train_Target, epochs=10, batch_size=32, validation_split=0.2)
    Pred = model.predict(Test_Data)
    Eval = evaluation(Pred,Test_Target)
    return Eval
