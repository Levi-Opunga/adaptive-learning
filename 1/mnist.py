import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Download and prepare the MNIST dataset
def load_mnist_data():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape images to include channel dimension
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


# Create a Convolutional Neural Network model
def create_cnn_model():
    model = keras.Sequential([
        # Convolutional layers
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # Flatten and add dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Train the model
def train_model(model, x_train, y_train, x_test, y_test):
    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

    return history


# Generate and plot confusion matrix
def plot_confusion_matrix(model, x_test, y_test):
    # Predict classes
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for MNIST Digit Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


# Visualize some predictions
def visualize_predictions(model, x_test, y_test):
    # Predict classes
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Plot some examples
    plt.figure(figsize=(15, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'True: {y_test[i]}, Pred: {y_pred_classes[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Main execution
def main():
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Create and train the model
    model = create_cnn_model()
    history = train_model(model, x_train, y_train, x_test, y_test)

    # Plot confusion matrix
    plot_confusion_matrix(model, x_test, y_test)

    # Visualize some predictions
    visualize_predictions(model, x_test, y_test)


# Run the main function
if __name__ == '__main__':
    main()
