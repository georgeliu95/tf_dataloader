import tensorflow as tf
import horovod.tensorflow as hvd
import nvtx
import sys


METHOD_ID = 1
if(len(sys.argv)==2 and int(sys.argv[1])<4):
    METHOD_ID = int(sys.argv[1])
    print("[EXAMPLE]\tMETHOD_ID=", METHOD_ID)
else:
    print("[EXAMPLE]\tTake METHOD_ID=1 as default")


# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


print('Device: {}'.format(hvd.rank()))

rng_0 = nvtx.start_range(message="model")
model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(4096, activation='sigmoid'),
    tf.keras.layers.Dense(1024, activation='sigmoid'),
    tf.keras.layers.Dense(256, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

opt = tf.optimizers.Adam(0.001 * hvd.size())
opt = hvd.DistributedOptimizer(opt)
nvtx.end_range(rng_0)


STEPS = 50
GLOBAL_BATCH_SIZE = 65536


rng_1 = nvtx.start_range(message="random data")
features = tf.random.normal([GLOBAL_BATCH_SIZE, 8 * 1024], dtype=tf.float32)
labels = tf.random.uniform([GLOBAL_BATCH_SIZE, 1], 0, 2, dtype=tf.int64)
nvtx.end_range(rng_1)


rng_2 = nvtx.start_range(message="dataset")
if(METHOD_ID==1):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).repeat(STEPS).batch(GLOBAL_BATCH_SIZE)
elif(METHOD_ID==2):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).shard(num_shards=hvd.size(), index=hvd.rank()).repeat(STEPS).batch(GLOBAL_BATCH_SIZE)
elif(METHOD_ID==3):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).shard(num_shards=hvd.size(), index=hvd.rank()).repeat(STEPS).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)




loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,  reduction=tf.keras.losses.Reduction.NONE)
nvtx.end_range(rng_2)


def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

@tf.function
def train_step(inputs, first_batch):
    features, labels = inputs

    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = compute_loss(labels, predictions)
    
    tape = hvd.DistributedGradientTape(tape)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)
    return loss


rng_3 = nvtx.start_range(message="training_phase", color="blue")
for batch, inputs in enumerate(dataset.take(STEPS // hvd.size())):
    loss = train_step(inputs, batch==0)

    if batch % 10 == 0 and hvd.local_rank() == 0:
        print('Step #%d\tLoss: %.6f' % (batch, loss))

nvtx.end_range(rng_3)
