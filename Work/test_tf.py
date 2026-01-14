import tensorflow as tf
import time

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
print(f"GPUs available: {gpus}")

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Create dataset
batch_size = 64
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(1000).batch(batch_size)

# Simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Optimizer
optimizer = tf.keras.optimizers.Adam(0.001)

# Training loop
print("\nStarting training...")
start_time = time.time()

num_epochs = 10
steps_per_epoch = len(x_train) // batch_size

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    running_loss = 0.0
    step = 0
    
    for batch_x, batch_y in dataset:
        with tf.GradientTape() as tape:
            logits = model(batch_x, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    batch_y, logits, from_logits=True
                )
            )
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        running_loss += loss.numpy()
        step += 1
        
        if step % 20 == 0:
            print(f'Step {step}/{steps_per_epoch}, Loss: {running_loss/20:.4f}')
            running_loss = 0.0

print(f"\nTotal training time: {time.time() - start_time:.2f} seconds")