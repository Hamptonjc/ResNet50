
###################################
#   Author: Jonathan Hampton
#   November 2020
#   github.com/hamptonjc
###################################


class Config(object):
    
    ##################################
    #  Hyper Parameters
    ##################################

    # Learning rate of optimizer
    LEARNING_RATE = 1e-4

    # Weight decay of optimizer
    WEIGHT_DECAY = 0.0001

    # Momentum of optimizer
    MOMENTUM = 0.9

    # Number of training epochs
    EPOCHS = 100

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
    RUN_NAME = 'full run 2'

    # saved model directory
    SAVED_WEIGHTS_DIR = 'checkpoints'

    # Name of network to use (currently only have resnet50)
    NETWORK_NAME = 'resnet50'

    # directory to store logs
    LOG_DIR = 'logs'

    ################################
    #  Callbacks
    ################################
    
    # bool to Reduce LR On Plateau
    REDUCE_LR_ON_PLATEAU = True
    
    # Number of epochs to judge plateau on
    RLROP_PATIENCE = 10
    
    # max amount of change in loss for RLROP to reduce LR
    RLROP_THRESHOLD = 0.0001
    
    # factor that the learning rate is reduced by
    RLROP_FACTOR = 0.1
    
    # lowest that the learning rate can be reduced to
    RLROP_LR_MIN = 0.00001

