

##############################
#   Author: Jonathan Hampton
#   November 2020
#   github.com/hamptonjc
###############################



##################################################
#  Imports 
##################################################

from typing import Tuple, NamedTuple, List
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers as kl
from config import Config as config

#############################################################################
#  ResNet-50
#############################################################################

class ResNet50(tf.keras.Model):

    # Type Handling
    class FPN_Modules(object):

        def __init__(self):
            super().__init__()

        class ConvLayers(NamedTuple):
            conv1: tf.keras.Model
            conv2: tf.keras.Model
            conv3: tf.keras.Model
            conv4: tf.keras.Model
            conv5: tf.keras.Model



    ################################################################
    #  Residual Block
    ################################################################

    class ResidualBlock(tf.keras.Model):
        """
        TF model class for ResNet50 residual blocks.
        init arguments correspond to parameters that change
                                            throughout the resnet architechture.

        """
        def __init__(self, name:str, input_shape:tuple, conv1_2_filters: int, conv3_filters: int,
                        conv1_stride: int=1, conv2_3_stride: int=1,
                        use_bias: bool=True, skip_connection_stride: int=1):
            super().__init__()
            self._name = name
            # Layers
            self.conv1 = kl.Conv2D(conv1_2_filters, 1, conv1_stride,
                                    padding='same', use_bias=use_bias,
                                    kernel_initializer=tf.random_normal_initializer(),
                                    name=name+'_conv1')

            self.conv2 = kl.Conv2D(conv1_2_filters, 3, conv2_3_stride,
                                    padding='same', use_bias=use_bias,
                                    kernel_initializer=tf.random_normal_initializer(),
                                    name=name+'_conv2')

            self.conv3 = kl.Conv2D(conv3_filters, 1, conv2_3_stride,
                                    padding='same', use_bias=use_bias,
                                    kernel_initializer=tf.random_normal_initializer(),
                                    name=name+'_conv3')
            self.bn1 = kl.BatchNormalization(name=name+'_bn1')
            self.bn2 = kl.BatchNormalization(name=name+'_bn2')
            self.bn3 = kl.BatchNormalization(name=name+'_bn3')

            self.skip_conv = kl.Conv2D(conv3_filters, 1, strides=skip_connection_stride,
                                        name=name+'_skip_connection_conv')

            self.build(input_shape)

        def call(self, input_tensor: tf.Tensor,
                    training: bool=None) -> tf.Tensor:
            skip_connection = self.skip_conv(input_tensor)
            x = self.bn1(input_tensor, training=training)
            x = tf.nn.relu(x)
            x = self.conv1(x)
            x = self.bn2(x, training=training)
            x = tf.nn.relu(x)
            x = self.conv2(x)
            x = self.bn3(x, training=training)
            x = tf.nn.relu(x)
            x = self.conv3(x)
            output = x + skip_connection
            return output




    def __init__(self, n_classes: int=config.N_CLASSES):

        super().__init__()
        
        # Misc. Parameters
        self.n_classes = n_classes

        # Training Functions
        self.loss_function = tf.keras.losses.CategoricalCrossentropy()
        self.optim = tfa.optimizers.SGDW(weight_decay=config.SGD_WEIGHT_DECAY,
                                        learning_rate=config.SGD_LEARNING_RATE,
                                        momentum=config.SGD_MOMENTUM)

        # Metric
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss_metric = tf.keras.metrics.Mean(name='validation_loss')
        self.train_acc_metric = tf.keras.metrics.Mean(name='training_accuracy')
        self.val_acc_metric = tf.keras.metrics.Mean(name='validation_accuracy')

        ########## Layers ############
        # Stage I
        self.conv1 = kl.Conv2D(64, 7, strides=2, padding='same',name='conv1_conv',
                                kernel_initializer=tf.random_normal_initializer())
        self.conv1_bn = kl.BatchNormalization(name='conv1_bn')
        self.conv1_relu = kl.ReLU(name='conv1_relu')
        self.pool1 = kl.MaxPooling2D(pool_size=(3,3), strides=(2,2),
                                        padding='same',name='conv1_pool')

        # Stage II
        self.conv2_1 = self.ResidualBlock(input_shape=(None,56,56,64),conv1_2_filters=64,
                                            conv3_filters=256,conv1_stride=1, conv2_3_stride=1,
                                            name='conv2_block1')

        self.conv2_2 = self.ResidualBlock(input_shape=(None,56,56,256),conv1_2_filters=64,
                                            conv3_filters=256,conv1_stride=1, conv2_3_stride=1,
                                            name='conv2_block2')

        self.conv2_3 = self.ResidualBlock(input_shape=(None,56,56,256), conv1_2_filters=64,
                                            conv3_filters=256,conv1_stride=1, conv2_3_stride=1,
                                            name='conv2_block3')

        # Stage III
        self.conv3_1 = self.ResidualBlock(input_shape=(None,28,28,256), conv1_2_filters=128,
                                            conv3_filters=512, conv1_stride=2,
                                            conv2_3_stride=1,skip_connection_stride=2,
                                            name='conv3_block1')

        self.conv3_2 = self.ResidualBlock(input_shape=(None,28,28,512), conv1_2_filters=128,
                                            conv3_filters=512,conv1_stride=1, conv2_3_stride=1,
                                            name='conv3_block2')

        self.conv3_3 = self.ResidualBlock(input_shape=(None,28,28,512), conv1_2_filters=128,
                                            conv3_filters=512, conv1_stride=1, conv2_3_stride=1,
                                            name='conv3_block3')

        self.conv3_4 = self.ResidualBlock(input_shape=(None,28,28,512), conv1_2_filters=128,
                                            conv3_filters=512, conv1_stride=1, conv2_3_stride=1,
                                            name='conv3_block4')

        # Stage IV
        self.conv4_1 = self.ResidualBlock(input_shape=(None,14,14,512), conv1_2_filters=256,
                                            conv3_filters=1024, conv1_stride=2,
                                            conv2_3_stride=1, skip_connection_stride=2,
                                            name='conv4_block1')

        self.conv4_2 = self.ResidualBlock(input_shape=(None,14,14,1024), conv1_2_filters=256,
                                            conv3_filters=1024, conv1_stride=1, conv2_3_stride=1,
                                            name='conv4_block2')

        self.conv4_3 = self.ResidualBlock(input_shape=(None,14,14,1024), conv1_2_filters=256,
                                            conv3_filters=1024, conv1_stride=1, conv2_3_stride=1,
                                            name='conv4_block3')

        self.conv4_4 = self.ResidualBlock(input_shape=(None,14,14,1024), conv1_2_filters=256,
                                            conv3_filters=1024, conv1_stride=1, conv2_3_stride=1,
                                            name='conv4_block4')

        self.conv4_5 = self.ResidualBlock(input_shape=(None,14,14,1024), conv1_2_filters=256,
                                            conv3_filters=1024, conv1_stride=1, conv2_3_stride=1,
                                            name='conv4_block5')

        self.conv4_6 = self.ResidualBlock(input_shape=(None,14,14,1024),conv1_2_filters=256,
                                            conv3_filters=1024, conv1_stride=1, conv2_3_stride=1,
                                            name='conv4_block6')

        # Stage V
        self.conv5_1 = self.ResidualBlock(input_shape=(None,7,7,1024), conv1_2_filters=512,
                                            conv3_filters=2048, conv1_stride=2,
                                            conv2_3_stride=1, skip_connection_stride=2,
                                            name='conv5_block1')

        self.conv5_2 = self.ResidualBlock(input_shape=(None,7,7,2048), conv1_2_filters=512,
                                            conv3_filters=2048, conv1_stride=1, conv2_3_stride=1,
                                            name='conv5_block2')

        self.conv5_3 = self.ResidualBlock(input_shape=(None,7,7,2048), conv1_2_filters=512,
                                            conv3_filters=2048, conv1_stride=1, conv2_3_stride=1,
                                            name='conv5_block3')


        self.final_pooling = kl.GlobalAveragePooling2D(name='final_pooling')
        self.final_fc = kl.Dense(self.n_classes, activation='softmax',name='final_fc')

#    def call(self, input_tensor):
#        x = self.conv1(input_tensor)
#        x = self.conv1_bn(x)
#        x = self.conv1_relu(x)
#        x = self.pool1(x)
#
#        #Stage II
#        x = self.conv2_1(x)
#        x = self.conv2_2(x)
#        x = self.conv2_3(x)
#
#        # Stage III
#        x = self.conv3_1(x)
#        x = self.conv3_2(x)
#        x = self.conv3_3(x)
#        x = self.conv3_4(x)
#
#        # Stage IV
#        x = self.conv4_1(x)
#        x = self.conv4_2(x)
#        x = self.conv4_3(x)
#        x = self.conv4_4(x)
#        x = self.conv4_5(x)
#        x = self.conv4_6(x)
#
#        # Stage V
#        x = self.conv5_1(x)
#        x = self.conv5_2(x)
#        x = self.conv5_3(x)
#
#        x = self.final_pooling(x)
#        x = self.final_fc(x)
#        return tf.cast(x, tf.float32)
#       
    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv1_bn(x)
        x = self.conv1_relu(x)
        x = self.pool1(x)

        # Stage II
        for layer in self.conv2_1.layers:
            x = layer(x)

        for layer in self.conv2_2.layers:
            x = layer(x)
        
        for layer in self.conv2_3.layers:
            x = layer(x)

        # Stage III
        for layer in self.conv3_1.layers:
            x = layer(x)

        for layer in self.conv3_2.layers:
            x = layer(x)

        for layer in self.conv3_3.layers:
            x = layer(x)

        for layer in self.conv3_4.layers:
            x = layer(x)

        # Stage IV
        for layer in self.conv4_1.layers:
            x = layer(x)

        for layer in self.conv4_2.layers:
            x = layer(x)

        for layer in self.conv4_3.layers:
            x = layer(x)

        for layer in self.conv4_4.layers:
            x = layer(x)

        for layer in self.conv4_5.layers:
            x = layer(x)

        for layer in self.conv4_6.layers:
            x = layer(x)
        
        # Stage V
        for layer in self.conv5_1.layers:
            x = layer(x)

        for layer in self.conv5_2.layers:
            x = layer(x)

        for layer in self.conv5_3.layers:
            x = layer(x)

        
        x = self.final_pooling(x)
        x = self.final_fc(x)
        return tf.cast(x, tf.float32)


