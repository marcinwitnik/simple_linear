import tensorflow as tf
import numpy as np

x_train = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float).reshape(-1, 1)
y_train = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
])


model.compile(optimizer='sgd', loss='mean_squared_error')


model.fit(x_train, y_train, epochs=500, verbose=0)

new_x = np.array([[10.0]])
prediction = model.predict(new_x)
print(f"Predykcja dla x={new_x[0][0]}: y={prediction[0][0]:.2f}")





















