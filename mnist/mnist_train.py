import tensorflow as tf
import os

def load_data():
    return tf.keras.datasets.mnist.load_data()

def normalize_data(x_train, x_test):
    return x_train / 255.0, x_test / 255.0

def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def compile_model(model):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=5, batch_size=128)

def evaluate_model(model, x_test, y_test):
    return model.evaluate(x_test, y_test)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = normalize_data(x_train, x_test)

model = create_model()
compile_model(model)
train_model(model, x_train, y_train)
evaluate_model(model, x_test, y_test)
save_model(model, './mnist/model.keras')
