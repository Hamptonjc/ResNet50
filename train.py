

########################################
#   Author: Jonathan Hampton
#   November 2020
#   github.com/hamptonjc  
########################################



#######################################
#  Imports
#######################################

import os
from termcolor import colored
import tensorflow as tf
import tensorflow_datasets as tfds
from ResNet50 import ResNet50
from config import Config as config



######################################################################
# Trainer Class
######################################################################

class Trainer:

    def __init__(self, network: str=config.NETWORK_NAME, data_dir=config.DATA_DIR):

        if network == 'resnet50':
            self.network = ResNet50()

        else:
            raise ValueError("resnet50 is the only network available currently")

        # Hyperparameters
        self.epochs = config.EPOCHS
        
        # Data
        self.data_dir = config.DATA_DIR
        imagenet_builder = tfds.builder("imagenet2012", data_dir=self.data_dir)
        data_dl_config = tfds.download.DownloadConfig(extract_dir=self.data_dir,
                                                        manual_dir=self.data_dir)
        imagenet_builder.download_and_prepare(download_dir=self.data_dir,
                                                download_config=data_dl_config)
        self.train_ds = tfds.load('imagenet2012',data_dir=self.data_dir, split='train')
        self.val_ds = tfds.load('imagenet2012',data_dir=self.data_dir, split='validation')
        
        # Misc.
        self.accuracy_func = tf.keras.metrics.Accuracy()


    def train(self, run_name, batch_size, epochs):
        train_ds = self.train_ds.take(1).map(self.data_preprocessing)
        val_ds = train_ds
        #val_ds = self.val_ds.take(1).map(self.data_preprocessing)
        self.training_loop(self.network, run_name=run_name, epochs=epochs,
                            train_dataset=train_ds.batch(batch_size),
                            validation_dataset=val_ds.batch(batch_size))



    ''' Training functions '''
    @tf.function
    def training_step(self, model, image, label):
        with tf.GradientTape() as tape:
            pred = model(image)
            loss = model.loss_function(label, pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            tf.print('TRAINING LOSS:', tf.math.reduce_mean(loss), end="\r")

        accuracy = self.accuracy_func(label, pred)
        model.optim.apply_gradients((zip(gradients, model.trainable_variables)))
        model.train_loss_metric(loss)
        model.train_acc_metric(accuracy)

    @tf.function
    def validation_step(self, model, image, label):
        pred = model(image)
        loss = model.loss_function(label, pred)
        tf.print('VALIDATION LOSS:', tf.math.reduce_mean(loss), end="\r")
        accuracy = self.accuracy_func(label, pred)
        model.val_loss_metric(loss)
        model.val_acc_metric(accuracy)


    def training_loop(self, model, epochs, train_dataset, validation_dataset, run_name):

        ##### Log Setup #####
        train_summary_writer, val_summary_writer = self.setup_logs(run_name)
        if not os.path.exists(f'{config.SAVED_WEIGHTS_DIR}/{run_name}/'):
            os.makedirs(f'{config.SAVED_WEIGHTS_DIR}/{run_name}')
        
        ##### CallBack Initalizations #####
        if config.REDUCE_LR_ON_PLATEAU:
            val_loss = []


        ####### Main Loop ###########
        for epoch in range(epochs):

            ###### Initialization ######
            model.train_loss_metric.reset_states()
            model.val_loss_metric.reset_states()
            model.train_acc_metric.reset_states()
            model.val_acc_metric.reset_states()

            ##### Training ######
            print(colored(f"\n\n================= Epoch {epoch+1} ====================","cyan",
                    attrs=['bold']))
            print("\n\nTraining Step Beginning...\n\n")
            for example in train_dataset:
                image, label = example["image"], example["label"]
                image, label = tf.cast(image, tf.float32), tf.cast(label, tf.float32)
                self.training_step(model, image, label)
            with train_summary_writer.as_default():
                tf.summary.scalar('train loss', model.train_loss_metric.result(), step=epoch+1)
                tf.summary.scalar('train accuracy', model.train_acc_metric.result(), step=epoch+1)
            


            ###### Validation #######
            print(f"\n\nEpoch {epoch+1} Training Complete! \n\nValidation Beginning...")
            for example in validation_dataset:
                image, label = example["image"], example["label"]
                image, label = tf.cast(image, tf.float32), tf.cast(label, tf.float32)
                self.validation_step(model, image, label)
            with val_summary_writer.as_default():
                tf.summary.scalar('validation loss', model.val_loss_metric.result(), step=epoch+1)
                tf.summary.scalar('validation accuracy', model.val_acc_metric.result(),step=epoch+1)
           
           ##### Save Model Weights #####
            model.save_weights(f'{config.SAVED_WEIGHTS_DIR}/{run_name}'+
                                f'/epoch_{epoch}_loss={model.val_loss_metric.result()}.h5')
            
            ###### Callbacks ######

            # Reduce LR On Plateau
            if config.REDUCE_LR_ON_PLATEAU:
                val_loss.append(model.val_loss_metric.result())
                val_loss = self.ReduceLROnPlateau(val_loss)


            ###### Epoch Summary #####
            print(f'\n\nEpoch {epoch+1} Complete!'+
                f'\n\nAverage Training Loss: {model.train_loss_metric.result()}'+
                f'\nAverage Validation Loss: {model.val_loss_metric.result()}')
    

    def setup_logs(self, run_name):
        train_log_dir = 'logs/' + run_name + '/train'
        val_log_dir = 'logs/' + run_name + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        return train_summary_writer, val_summary_writer


    def data_preprocessing(self, example):
        image, label = example["image"], example["label"]
        image = tf.image.resize(image, config.INPUT_IMAGE_SIZE)
        label = tf.one_hot(label, self.network.n_classes)
        image = tf.image.per_image_standardization(image)
        example["image"], example["label"] = image, label
        return example

    
    def ReduceLROnPlateau(self, monitor):
        if len(monitor) < config.RLROP_PATIENCE:
            return monitor

        if len(monitor) > config.RLROP_PATIENCE:
            monitor.pop(0)

        if monitor[0] - monitor[-1] < config.RLROP_THRESHOLD:
            self.network.optim.lr.assign(self.network.optim.lr.numpy().item()*config.RLROP_FACTOR)
            print(colored(f"\n\nCallback ReduceLROnPlateau reduced the learning rate " +
                    f"to {self.network.optim.lr.numpy().item()}", 'yellow',attrs=['underline']))
            monitor = []

        return monitor



if __name__ == '__main__':
    res_trainer = Trainer()
    res_trainer.train(run_name=config.RUN_NAME,
                        batch_size=config.BATCH_SIZE,
                        epochs=config.EPOCHS)
    
    print("\n\n Training Complete! :-)")


