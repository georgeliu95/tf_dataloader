import tensorflow as tf
import nvtx

mirrored_strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

STEPS = 50
GLOBAL_BATCH_SIZE = 65536

rng = nvtx.start_range(message="data", color="blue")
# FillPhiloxRandomKernelLaunch & EigenMetaKernel on GPU：0 as default
features = tf.random.normal([GLOBAL_BATCH_SIZE, 8 * 1024], dtype=tf.float32)
labels = tf.random.uniform([GLOBAL_BATCH_SIZE, 1], 0, 2, dtype=tf.int64)
print(features.device)
print(labels.device)
nvtx.end_range(rng)
# DtoHAsync
# Dataset is on CPU
dataset = tf.data.Dataset.from_tensor_slices((features, labels)).repeat(STEPS).batch(GLOBAL_BATCH_SIZE)

rng1 = nvtx.start_range(message="model", color="yellow")
with mirrored_strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4096, activation='sigmoid'),
        tf.keras.layers.Dense(1024, activation='sigmoid'),
        tf.keras.layers.Dense(256, activation='sigmoid'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])


class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))
nvtx.end_range(rng1)

# on each GPU
rng2 = nvtx.start_range(message="dist_dataset", color="red")
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
for id,it in enumerate(dist_dataset):
    if id==0:
        [print(x) for x in it]
nvtx.end_range(rng2)
