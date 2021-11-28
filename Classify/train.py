from utils.models.Custom import ModelTraining
import os
import tensorflow as tf 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


trainer = ModelTraining()
trainer.setModelTypeAsInceptionV3()
trainer.setDataDirectory("data")
trainer.trainModel(num_objects=123, num_experiments=1000, enhance_data=True, batch_size=4, show_network_summary=True, continue_from_model="model_ex-004_acc-0.987975.h5", initial_num_objects=1000)