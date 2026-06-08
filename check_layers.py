import tensorflow as tf

model = tf.keras.models.load_model("brain_tumor_model.keras")

for layer in model.layers:
    print(layer.name)