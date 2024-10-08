from tensorflow.keras import mixed_precision
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

mixed_precision.set_global_policy('mixed_float16')

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

inputs = keras.Input(shape=(784,), name='digits')
if tf.config.list_physical_devices('GPU'):
    print('The model will run with 4096 units on a GPU')
    num_units = 500
else:
    # Use fewer units on CPUs so the model finishes in a reasonable amount of time
    print('The model will run with 64 units on a CPU')
    num_units = 64
dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
x = dense1(inputs)
dense2 = layers.Dense(num_units, activation='relu', name='dense_2')
x = dense2(x)

print(dense1.dtype_policy)
print('x.dtype: %s' % x.dtype.name)
# 'kernel' is dense1's variable
print('dense1.kernel.dtype: %s' % dense1.kernel.dtype.name)

# INCORRECT: softmax and model output will be float16, when it should be float32
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)

# CORRECT: softmax and model output are float32
x = layers.Dense(10, name='dense_logits')(x)
outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)

outputs = layers.Activation('linear', dtype='float32')(outputs)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

initial_weights = model.get_weights()

history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
