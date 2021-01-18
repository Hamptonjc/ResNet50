

###############################
#  Imports
###############################
import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
from ResNet50 import ResNet50
from data import ImageNet
from config import Config


#######################################################
#  Main Script
#######################################################

def main()->None:
    
    # Data
    imagenet = ImageNet(take=10)
    train_ds, val_ds = imagenet.train_ds, imagenet.val_ds


    # Callbacks
    callbacks = []
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=Config.SAVED_WEIGHTS_DIR,
                        monitor='val_loss', save_best_only=True, mode='min'))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=Config.LOG_DIR+'/'+ \
                                                    Config.RUN_NAME))

    # Model
    model = ResNet50()
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer='Adam', loss=loss_function)
    
    # Train
    model.fit(train_ds,
                validation_data = train_ds,
                epochs=Config.EPOCHS,
                callbacks=callbacks)


if __name__=='__main__':
    main()



