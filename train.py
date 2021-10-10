# Importing the required libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
# from keras.regularizers import l1, l2
import matplotlib.pyplot as plt 
import numpy as np 



def plot_training(summary):
	acc = summary.history['accuracy']
	val_acc = summary.history['val_accuracy']
	epochs = range(len(acc))

	plt.figure()
	plt.plot(epochs, acc, 'r-')
	plt.plot(epochs, val_acc, 'g-')
	plt.title('Training and validation accuracy')
	plt.show()
	plt.savefig('summary.png')

def plot_validation(summary):
	loss = summary.history['loss']
	val_loss = summary.history['val_loss']
	epochs = range(len(loss))

	plt.figure()
	plt.plot(epochs, loss, 'r-')
	plt.plot(epochs, val_loss, 'g-')
	plt.title('Training and validation loss')
	plt.show()
	plt.savefig('summary.png')


train_path = './train_images/'
validation_path = './test_images/'



BATCH_SIZE = 16
epochs = 50
HEIGHT = 224
WIDTH = 224



train_aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")



validation_aug = ImageDataGenerator()



mean = np.array([123.68, 116.779, 103.939], dtype="float32")
train_aug.mean = mean
validation_aug.mean = mean



train_gen = train_aug.flow_from_directory(
	train_path,
	class_mode="categorical",
	target_size=(HEIGHT, WIDTH),
	color_mode="rgb",
	shuffle=True,
	batch_size=BATCH_SIZE)



validation_gen = validation_aug.flow_from_directory(
	validation_path,
	class_mode="categorical",
	target_size=(HEIGHT, WIDTH),
	color_mode="rgb",
	shuffle=False,
	batch_size=BATCH_SIZE)


baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(HEIGHT, WIDTH, 3)))


for layer in baseModel.layers:
    layer.trainable = False


headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(1024, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(50, activation="softmax")(headModel)


model = Model(inputs=baseModel.input, outputs=headModel)

rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
opt = Adam(learning_rate=0.001)

model.compile(loss="categorical_crossentropy", 
	optimizer=opt, metrics=["accuracy"])

print('***** MODEL COMPILED *****')


print('***** TRAINING MODEL *****')
history = model.fit_generator(
	train_gen,
	steps_per_epoch=len(train_gen) // BATCH_SIZE,
	validation_data=validation_gen,
	validation_steps=len(validation_gen) // BATCH_SIZE,
	epochs=epochs,
	callbacks=[rlrop])



model.save('model.h5')
print('***** MODEL SAVED *****')

plot_training(history)
plot_validation(history)