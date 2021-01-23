
############################################
#  Imports
############################################
import tensorflow as tf
import tensorflow_datasets as tfds
from config import Config

#########################################################################
#  ImageNet 2012
#########################################################################

class ImageNet:

    def __init__(self, batch_size: int=Config.BATCH_SIZE,
                    data_dir: str=Config.DATA_DIR,
                    take:int=None)->None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        imagenet_builder = tfds.builder("imagenet2012", data_dir=self.data_dir)
        data_dl_config = tfds.download.DownloadConfig(extract_dir=self.data_dir,
                                                        manual_dir=self.data_dir)
        imagenet_builder.download_and_prepare(download_dir=self.data_dir,
                                                download_config=data_dl_config)
        self.train_ds = tfds.load('imagenet2012',data_dir=self.data_dir, split='train')
        self.val_ds = tfds.load('imagenet2012',data_dir=self.data_dir, split='validation')
        self.train_ds = self.train_ds.map(self._data_preprocessing).batch(batch_size)
        self.val_ds = self.val_ds.map(self._data_preprocessing).batch(batch_size)
        
        if take is not None:
            self.train_ds = self.train_ds.take(take)
            self.val_ds = self.val_ds.take(take)


    def _data_preprocessing(self, example: tf.train.Example)->tf.train.Example:
        image, label = example["image"], example["label"]
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, Config.INPUT_IMAGE_SIZE)
        image = tf.keras.applications.resnet_v2.preprocess_input(image)
        label = tf.expand_dims(label, axis=0)
        label = tf.cast(label, tf.float32)
        example["image"], example["label"] = image, label
        return example

