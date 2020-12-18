import tensorflow as tf
from ResNet50 import ResNet50


if __name__ == '__main__':
    res = ResNet50()
    #a = tf.ones((1,224,224,3))
    #print(res(a))
    print(res.summary())

