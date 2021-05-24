import tensorflow as tf
import horovod.tensorflow as hvd
import nvtx
import sys


METHOD_ID = 1
if(len(sys.argv)==2 and int(sys.argv[1])<10):
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


mirrored_strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

rng_0 = nvtx.start_range(message="model")
with mirrored_strategy.scope():
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

    # optimizer = tf.keras.optimizers.Adam()
nvtx.end_range(rng_0)


STEPS = 50
GLOBAL_BATCH_SIZE = 65536


rng_1 = nvtx.start_range(message="random data")
features = tf.random.normal([GLOBAL_BATCH_SIZE, 8 * 1024], dtype=tf.float32)
labels = tf.random.uniform([GLOBAL_BATCH_SIZE, 1], 0, 2, dtype=tf.int64)
nvtx.end_range(rng_1)


rng_2 = nvtx.start_range(message="dataset")
dataset, dist_dataset = None, None
if(METHOD_ID in [1,2]):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).repeat(STEPS).batch(GLOBAL_BATCH_SIZE)
    if(METHOD_ID==2):
        dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

elif(METHOD_ID==3):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).repeat(STEPS).batch(GLOBAL_BATCH_SIZE).prefetch(5)
    dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

elif(METHOD_ID in [4,5,6,7]):
    def dataset_fn(input_context):
        batch_size = input_context.get_per_replica_batch_size(GLOBAL_BATCH_SIZE)
        features = tf.random.normal([batch_size, 8 * 1024], dtype=tf.float32)
        labels = tf.random.uniform([batch_size, 1], 0, 2, dtype=tf.int64)
        if(METHOD_ID in [4,6]):
            dataset = tf.data.Dataset.from_tensor_slices((features, labels)).repeat(STEPS).batch(batch_size)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((features, labels)).repeat(STEPS).batch(batch_size).prefetch(5)
        return dataset
    
    if(METHOD_ID in [4,5]):
        dist_dataset = mirrored_strategy.distribute_datasets_from_function(dataset_fn, tf.distribute.InputOptions(
                        experimental_replication_mode = tf.distribute.InputReplicationMode.PER_REPLICA))
    else:
        dist_dataset = mirrored_strategy.distribute_datasets_from_function(dataset_fn, tf.distribute.InputOptions(
                        experimental_prefetch_to_device = False,
                        experimental_replication_mode = tf.distribute.InputReplicationMode.PER_REPLICA,
                        experimental_place_dataset_on_device = True))
elif(METHOD_ID in [8,9]):
    def feature_value_fn(ctx):
        batch_size = GLOBAL_BATCH_SIZE // mirrored_strategy.num_replicas_in_sync
        if(METHOD_ID==9):
            with tf.device('GPU:{}'.format(ctx.replica_id_in_sync_group)):
                features = tf.random.normal([batch_size, 8 * 1024], dtype=tf.float32)
        else:
            features = tf.random.normal([batch_size, 8 * 1024], dtype=tf.float32)
        return features

    def label_value_fn(ctx):
        batch_size = GLOBAL_BATCH_SIZE // mirrored_strategy.num_replicas_in_sync
        if(METHOD_ID==9):
            with tf.device('GPU:{}'.format(ctx.replica_id_in_sync_group)):
                labels = tf.random.uniform([batch_size, 1], 0, 2, dtype=tf.int64)
        else:
            labels = tf.random.uniform([batch_size, 1], 0, 2, dtype=tf.int64)
        return labels

    dist_feature_values = mirrored_strategy.experimental_distribute_values_from_function(feature_value_fn)
    dist_label_values = mirrored_strategy.experimental_distribute_values_from_function(label_value_fn)


dataset = dataset.shard(hvd.size(), hvd.rank())
if dist_dataset is not None:
    dist_dataset = dist_dataset.shard(hvd.size(), hvd.rank())


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,  reduction=tf.keras.losses.Reduction.NONE)
nvtx.end_range(rng_2)


def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

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

@tf.function
def distributed_train_step(inputs, first_batch):
    per_replica_losses = mirrored_strategy.run(train_step, args=(inputs, first_batch))
    return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


rng_3 = nvtx.start_range(message="training_phase", color="blue")
if(METHOD_ID < 8):
    for batch, inputs in enumerate(dataset // hvd.size()):
        print(distributed_train_step(inputs, batch==0))
else:
    for i in range(STEPS // hvd.size()):
        print(distributed_train_step((dist_feature_values, dist_label_values), i==0))
nvtx.end_range(rng_3)
