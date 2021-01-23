

###############################
#  Imports
###############################
import types
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import horovod.tensorflow as hvd
import horovod.keras as hvdK
from ResNet50 import ResNet50
from imagenet import ImageNet
from config import Config


#######################################################
#  Main Script
#######################################################

def main()->None:

    # Horovod Init
    hvd.init()
    size=hvd.size()

    # Config GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # get optimizer & loss function
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    opt = tf.keras.optimizers.Adam(lr=Config.LEARNING_RATE*size)

    # Data
    imagenet = ImageNet(take=20)
    train_ds, val_ds = imagenet.train_ds, imagenet.val_ds
    n_train_batches = train_ds.cardinality().numpy()
    n_val_batches = val_ds.cardinality().numpy()

    # Callbacks
    callbacks = []
    callbacks.append(hvdK.callbacks.BroadcastGlobalVariablesCallback(0))
    callbacks.append(hvdK.callbacks.MetricAverageCallback())
    callbacks.append(hvdK.callbacks.LearningRateWarmupCallback(warmup_epochs=5,
                        initial_lr=Config.LEARNING_RATE))
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))
    
    if hvd.rank() == 0:
        
        ckpt_dir = Config.SAVED_WEIGHTS_DIR + "/" + Config.RUN_NAME
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
                                
        ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_dir+ \
                                                    "/epoch-{epoch:02d}-loss={val_loss:.2f}.h5",
                                                    monitor='val_loss', save_best_only=True, mode='min')

        log_dir = Config.LOG_DIR + "/" + Config.RUN_NAME
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        callbacks.append(ckpt)
        callbacks.append(tensorboard)
        callbacks.append(tfa.callbacks.TQDMProgressBar())

    # Model
    model = ResNet50()
    model.loss_function = loss_function
    model.train_step = types.MethodType(distributed_train_step, model)
    model.compile(optimizer=opt, loss=loss_function)
    
    # Train
    model.fit(train_ds,
                steps_per_epoch = n_train_batches//size,
                validation_data = val_ds,
                validation_steps = n_val_batches//size,
                epochs=Config.EPOCHS,
                verbose = 0,
                callbacks=callbacks)



# Updated model train step for distributed training
def distributed_train_step(self, example: tf.train.Example)->dict:
    # Unpack data
    image, label = example["image"], example["label"]

    with tf.GradientTape() as tape:
        tape = hvd.DistributedGradientTape(tape)
        # Calculate prediction
        pred = self(image)
        # Calculate loss
        loss = self.loss(label, pred)
        # Compute gradients
    gradients = tape.gradient(loss, self.trainable_variables)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # Update metrics
    self.train_loss_metric(loss)
    self.train_top_1_metric(label, pred)
    self.train_top_5_metric(label, pred)
    return {"loss": self.train_loss_metric.result(),
            "accuracy": self.train_top_1_metric.result(),
            "top 5": self.train_top_5_metric.result()}



if __name__=='__main__':
    Config.RUN_NAME = input('Enter run name:')
    main()


