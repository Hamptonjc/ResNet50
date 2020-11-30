

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
        def __init__(self, name:str, input_shape:tuple,
                        conv1_filters:int, conv2_filters:int, conv3_filters: int,
                        conv1_stride:int, conv2_stride:int, conv3_stride:int,
                        skip_conv:bool=False, skip_pool:bool=False, skip_connection_stride: int=(1,1)):
            
            super().__init__() 
            self._name = name
            
            # Layers
            self.input_layer = kl.InputLayer(input_shape)
            self.preact_bn = kl.BatchNormalization(epsilon=1.001e-05, name=name+'_preact_bn')
            self.preact_relu = kl.Activation('relu', name=name+'_preact_relu')
            self.conv1 = kl.Conv2D(filters=conv1_filters, kernel_size=(1,1), strides=conv1_stride,
                                    use_bias=False, activation='linear', name=name+'_conv1')
            self.bn1 = kl.BatchNormalization(epsilon=1.001e-05, name=name+'_bn1')
            self.relu1 = kl.Activation('relu', name=name+'_relu1')
            self.pad1 = kl.ZeroPadding2D(padding=((1,1),(1,1)), name=name+'_pad1')
            self.conv2 = kl.Conv2D(filters=conv2_filters, kernel_size=(3,3), strides=conv2_stride,
                                    use_bias=False, activation='linear', name=name+'_conv2')
            self.bn2 = kl.BatchNormalization(epsilon=1.001e-05, name=name+'_bn2')
            self.relu2 = kl.Activation('relu', name=name+'_relu2')
            self.skip_conv = None
            self.skip_pool = None
            if skip_conv:
                self.skip_conv = kl.Conv2D(filters=conv3_filters, kernel_size=(1,1),
                                            strides=skip_connection_stride,
                                            name=name+'_skip_conv')
            if skip_pool:
                self.skip_pool = kl.MaxPooling2D(pool_size=(1,1), strides=(2,2),
                                                    padding='valid', name=name+"_skip_pool")

            self.conv3 = kl.Conv2D(filters=conv3_filters, kernel_size=(1,1),
                                    strides=conv3_stride,activation='linear', name=name+'_conv3')

            # Build
            self.build(input_shape)

        def call(self, input_tensor: tf.Tensor,
                    training: bool=None) -> tf.Tensor:
            input_tensor = self.input_layer(input_tensor)
            x = self.preact_bn(input_tensor)
            x = self.preact_relu(x)
            skip_connection = x
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.pad1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv3(x)

            if self.skip_conv is not None:
                skip_connection = self.skip_conv(skip_connection)
            
            elif self.skip_pool is not None:
                skip_connection = self.skip_pool(input_tensor)

            else:
                skip_connection = input_tensor

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
        self.input_layer = kl.InputLayer(config.INPUT_SHAPE)

        self.conv1_pad = kl.ZeroPadding2D(padding=((3,3),(3,3)), name='conv1_pad')

        self.conv1_conv = kl.Conv2D(filters=64,kernel_size=(7,7), strides=(2,2),
                                    padding='valid', activation='linear',name='conv1_conv')

        self.pool1_pad = kl.ZeroPadding2D(padding=((1,1),(1,1)), name='pool1_pad')
        
        self.pool1_pool = kl.MaxPooling2D(pool_size=(3,3), strides=(2,2),
                                        padding='valid',name='pool1_pool')

        # Stage II
        self.conv2_1 = self.ResidualBlock(name='conv2_block1', input_shape=(None,56,56,64),
                                            conv1_filters=64, conv2_filters=64, conv3_filters=256,
                                            conv1_stride=(1,1), conv2_stride=(1,1), conv3_stride=(1,1),
                                            skip_conv=True, skip_connection_stride=(1,1))

        self.conv2_2 = self.ResidualBlock(name='conv2_block2', input_shape=(None,56,56,256),
                                            conv1_filters=64, conv2_filters=64, conv3_filters=256,
                                            conv1_stride=(1,1),conv2_stride=(1,1), conv3_stride=(1,1))

        self.conv2_3 = self.ResidualBlock(name='conv2_block3', input_shape=(None,56,56,256),
                                            conv1_filters=64, conv2_filters=64, conv3_filters=256,
                                            conv1_stride=(1,1), conv2_stride=(2,2), conv3_stride=(1,1),
                                            skip_pool=True)
        # Stage III
        self.conv3_1 = self.ResidualBlock(name='conv3_block1', input_shape=(None,28,28,256),
                                            conv1_filters=128, conv2_filters=128, conv3_filters=512,
                                            conv1_stride=(1,1), conv2_stride=(1,1), conv3_stride=(1,1),
                                            skip_conv=True, skip_connection_stride=(1,1))
                                            

        self.conv3_2 = self.ResidualBlock(name='conv3_block2', input_shape=(None,28,28,512),
                                            conv1_filters=128, conv2_filters=128, conv3_filters=512,
                                            conv1_stride=(1,1), conv2_stride=(1,1), conv3_stride=(1,1))
                                            

        self.conv3_3 = self.ResidualBlock(name='conv3_block3', input_shape=(None,28,28,512),
                                            conv1_filters=128, conv2_filters=128, conv3_filters=512,
                                            conv1_stride=(1,1), conv2_stride=(1,1), conv3_stride=(1,1))
                                            

        self.conv3_4 = self.ResidualBlock(name='conv3_block4', input_shape=(None,28,28,512),
                                            conv1_filters=128, conv2_filters=128, conv3_filters=512,
                                            conv1_stride=(1,1), conv2_stride=(2,2),conv3_stride=(1,1),
                                            skip_pool=True)

        # Stage IV
        self.conv4_1 = self.ResidualBlock(name='conv4_block1', input_shape=(None,14,14,512),
                                            conv1_filters=256, conv2_filters=256, conv3_filters=1024,
                                            conv1_stride=(1,1), conv2_stride=(1,1), conv3_stride=(1,1),
                                            skip_conv=True, skip_connection_stride=(1,1))
                                            

        self.conv4_2 = self.ResidualBlock(name='conv4_block2', input_shape=(None,14,14,1024),
                                            conv1_filters=256, conv2_filters=256, conv3_filters=1024,
                                            conv1_stride=(1,1), conv2_stride=(1,1), conv3_stride=(1,1))
                                            

        self.conv4_3 = self.ResidualBlock(name='conv4_block3', input_shape=(None,14,14,1024),
                                            conv1_filters=256, conv2_filters=256, conv3_filters=1024,
                                            conv1_stride=(1,1), conv2_stride=(1,1), conv3_stride=(1,1))
                                            

        self.conv4_4 = self.ResidualBlock(name='conv4_block4', input_shape=(None,14,14,1024),
                                            conv1_filters=256, conv2_filters=256, conv3_filters=1024,
                                            conv1_stride=(1,1), conv2_stride=(1,1), conv3_stride=(1,1))
                                            

        self.conv4_5 = self.ResidualBlock(name='conv4_block5', input_shape=(None,14,14,1024),
                                            conv1_filters=256, conv2_filters=256, conv3_filters=1024,
                                            conv1_stride=(1,1), conv2_stride=(1,1), conv3_stride=(1,1))
                                            

        self.conv4_6 = self.ResidualBlock(name='conv4_block6', input_shape=(None,14,14,1024),
                                            conv1_filters=256, conv2_filters=256, conv3_filters=1024,
                                            conv1_stride=(1,1), conv2_stride=(2,2), conv3_stride=(1,1),
                                            skip_pool=True)
                                            

        # Stage V
        self.conv5_1 = self.ResidualBlock(name='conv5_block1', input_shape=(None,7,7,1024),
                                            conv1_filters=512, conv2_filters=512, conv3_filters=2048,
                                            conv1_stride=(1,1), conv2_stride=(1,1), conv3_stride=(1,1),
                                            skip_conv=True, skip_connection_stride=(1,1))
                                            

        self.conv5_2 = self.ResidualBlock(name='conv5_block2', input_shape=(None,7,7,2048),
                                            conv1_filters=512, conv2_filters=512, conv3_filters=2048,
                                            conv1_stride=(1,1), conv2_stride=(1,1), conv3_stride=(1,1))
                                            

        self.conv5_3 = self.ResidualBlock(name='conv5_block3', input_shape=(None,7,7,2048),
                                            conv1_filters=512, conv2_filters=512, conv3_filters=2048,
                                            conv1_stride=(1,1), conv2_stride=(1,1), conv3_stride=(1,1))
                                            

        # Post Stage
        self.post_bn = kl.BatchNormalization(epsilon=1.001e-05, name='post_bn')

        self.post_relu = kl.Activation('relu', name='post_relu')

        self.final_pooling = kl.GlobalAveragePooling2D(name='final_pooling')

        self.predictions = kl.Dense(self.n_classes, activation='softmax',name='predictions')

        # Build
        self.build(config.INPUT_SHAPE)

    def call(self, input_tensor):
        # Stage I
        x = self.input_layer(input_tensor)
        x = self.conv1_pad(x)
        x = self.conv1_conv(x)
        x = self.pool1_pad(x)
        x = self.pool1_pool(x)

        # Stage II
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        # Stage III
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        # Stage IV
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)

        # Stage V
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        
        # Post Stage
        x = self.post_bn(x)
        x = self.post_relu(x)
        x = self.final_pooling(x)
        x = self.predictions(x)
        return tf.cast(x, tf.float32)
       



#    def call(self, input_tensor):
#
#        # Stage I
#        x = self.input_layer(input_tensor)
#
#        x = self.conv1_pad(x)
#
#        x = self.conv1_conv(x)
#
#        x = self.pool1_pad(x)
#
#        x = self.pool1_pool(x)
#
#
#        # Stage II
#        for layer in self.conv2_1.layers:
#            x = layer(x)
#
#        for layer in self.conv2_2.layers:
#            x = layer(x)
#        
#        for layer in self.conv2_3.layers:
#            x = layer(x)
#
#        # Stage III
#        for layer in self.conv3_1.layers:
#            x = layer(x)
#
#        for layer in self.conv3_2.layers:
#            x = layer(x)
#
#        for layer in self.conv3_3.layers:
#            x = layer(x)
#
#        for layer in self.conv3_4.layers:
#            x = layer(x)
#
#        # Stage IV
#        for layer in self.conv4_1.layers:
#            x = layer(x)
#
#        for layer in self.conv4_2.layers:
#            x = layer(x)
#
#        for layer in self.conv4_3.layers:
#            x = layer(x)
#
#        for layer in self.conv4_4.layers:
#            x = layer(x)
#
#        for layer in self.conv4_5.layers:
#            x = layer(x)
#
#        for layer in self.conv4_6.layers:
#            x = layer(x)
#        
#        # Stage V
#        for layer in self.conv5_1.layers:
#            x = layer(x)
#
#        for layer in self.conv5_2.layers:
#            x = layer(x)
#
#        for layer in self.conv5_3.layers:
#            x = layer(x)
#
#        # Post Stage
#        x = self.post_bn(x)
#
#        x = self.post_relu(x)
#
#        x = self.final_pooling(x)
#
#        x = self.predictions(x)
#
#        return tf.cast(x, tf.float32)
#





