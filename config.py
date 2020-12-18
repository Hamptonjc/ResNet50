
###################################
#   Author: Jonathan Hampton
#   November 2020
#   github.com/hamptonjc
###################################





#################################################################
#  ResNet Config (as per the original ResNet paper)
#################################################################

class Config(object):
    
    ##################################
    #  Hyper Parameters
    ##################################

    # Learning rate of optimizer
    SGD_LEARNING_RATE = 0.1

    # Weight decay of optimizer
    SGD_WEIGHT_DECAY = 0.0001

    # Momentum of optimizer
    SGD_MOMENTUM = 0.9

    # Number of training epochs
    EPOCHS = 60

    ################################
    #  Data
    ################################

    # Number of classes
    N_CLASSES = 1000    

    # Batch size for training
    BATCH_SIZE = 48

    # Data Directory
    DATA_DIR = 'data/'
    

    # Size of input images (images in dataset are resized to this)
    INPUT_IMAGE_SIZE = (224,224)

    # Number of classes
    N_CLASSES = 1000

    # Network input shape
    INPUT_SHAPE = (None,224,224,3)

    ################################
    #  Misc.
    ################################

    # Name of training run
    RUN_NAME = 'new preprocessing'

    # saved model directory
    SAVED_WEIGHTS_DIR = 'saved_weights'

    # Name of network to use (currently only have resnet50)
    NETWORK_NAME = 'resnet50'

    ################################
    #  Callbacks
    ################################
    
    # bool to Reduce LR On Plateau
    REDUCE_LR_ON_PLATEAU = True
    
    # Number of epochs to judge plateau on
    RLROP_PATIENCE = 3
    
    # max amount of change in loss for RLROP to reduce LR
    RLROP_THRESHOLD = 1.0
    
    # factor that the learning rate is reduced by
    RLROP_FACTOR = 0.1


