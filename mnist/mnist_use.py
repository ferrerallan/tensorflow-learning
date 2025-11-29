import tensorflow as tf
import numpy as np
from PIL import Image

def load_model(path):
    return tf.keras.models.load_model(path)

def load_image(path):
    image = Image.open(path).convert('L').resize((28, 28))
    image = np.array(image)
    image = 255 - image
    image = image / 255.0
    return image.reshape(1, 28, 28)

def predict(model, image):
    prediction = model.predict(image)
    return tf.argmax(prediction, axis=1).numpy()[0]

model = load_model('./models/model.keras')
image = load_image('mnist_test.png')
predicted = predict(model, image)
print('Predicted:', predicted)
