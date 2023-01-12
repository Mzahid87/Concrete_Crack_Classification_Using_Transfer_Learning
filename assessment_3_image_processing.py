
#%%
# 1. Import necessary packages
import os
import pathlib 
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,metrics,callbacks,applications

#%%
# 2. Load the data
file_path = r"C:\Users\acer\Desktop\Assessment_3_MohammadZahid\concrete_dataset"

# 3. Data preparation
# 3.1 Define the path to the train and validation data folder
data_dir = pathlib.Path(file_path)

# 3.2 Define the batch size and image size
SEED = 32
IMG_SIZE = (160,160)

# 3.3 Load the data into tensorflow data set using the specific method

train_dataset = tf.keras.utils.image_dataset_from_directory(data_dir, shuffle = True, validation_split = 0.2, subset="training", seed = SEED, image_size = IMG_SIZE , batch_size = 10)
val_dataset = tf.keras.utils.image_dataset_from_directory(data_dir, shuffle = True, validation_split = 0.2, subset="validation", seed = SEED, image_size = IMG_SIZE , batch_size = 10)

#%%
# 4. Display some images as example
class_names = train_dataset.class_names

plt.figure(figsize=(10,10))
for images,labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')

#%%
# 5. Further split the validation dataset into validation-test split
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

#%%
# 6. Convert the BatchDataset into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = validation_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

#%%
# 7. Create a small pipeline for data augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#%%
# 7.1 Apply the data augmentation to test it out
for images,labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')

#%%
# 8. Prepare the layer for data preprocessing
preprocess_input = applications.mobilenet_v2.preprocess_input

# 9. Apply transfer learning
IMG_SHAPE = IMG_SIZE + (3,)
feature_extractor = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

# 9.1 Disable the training for the feature extractor (freeze the layers)
feature_extractor.trainable = False
feature_extractor.summary()
keras.utils.plot_model(feature_extractor,show_shapes=True)

#%%
# 10. Create the classification layers
global_avg = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(len(class_names),activation='softmax')

#%%
# 11. Use functional API to link all of the modules together
inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = feature_extractor(x)
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

#%%
# 12. Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

#%%
# 12.1 Evaluate the model before model training
loss0,accuracy0 = model.evaluate(pf_val)
print("Loss = ",loss0)
print("Accuracy = ",accuracy0)

#%%
# 12.2 TensorBoard callback
log_path = os.path.join('log_dir','assessment_3',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_dir=log_path)
#open anconda prompt and activate environment
#copy path dir above
#command --> tensorboard --logdir <path>
# -->"C:\Users\acer\Desktop\Assessment_3_MohammadZahid\log_dir\assessment_3" 
# then run the anaconda prompt
#open google to see the graph
#http://localhost:6006/

#%%
# 12.3 Train the model
EPOCHS = 3
history = model.fit(pf_train,validation_data=pf_val,epochs=EPOCHS,callbacks=[tb])

# %%
"""
Next, we are going to further fine tune the model by using a different transfer learning strategy --> fine tune pretrained model and frozen layers

What we are going to do is we are going to unfreeze some layers at the behind part of the feature extractor, so that those layers will be trained to extract the high features that we specifically want.
"""
# 13. Apply the next transfer learning strategy
feature_extractor.trainable = True

# 13.1 Freeze the earlier layers
for layer in feature_extractor.layers[:100]:
    layer.trainable = False

feature_extractor.summary()

#%%
# 14. Compile the model
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

#%%
# 15. Continue the model training with this new set of configuration
fine_tune_epoch = 3
total_epoch = EPOCHS + fine_tune_epoch

# 15.1 Follow up from the previous model training
history_fine = model.fit(pf_train,validation_data=pf_val,epochs=total_epoch,initial_epoch=history.epoch[-1],callbacks=[tb])

#%%
# 16. Evaluate the final model
test_loss,test_acc = model.evaluate(pf_test)

print("Loss = ",test_loss)
print("Accuracy = ",test_acc)

#%%
# 16.1 Deploy the model using the test data
image_batch, label_batch = pf_test.as_numpy_iterator().next()
predictions = np.argmax(model.predict(image_batch),axis=1)

# 16.2 Compare label and prediction
label_vs_prediction = np.transpose(np.vstack((label_batch,predictions)))

# %%
# 17. Save the modelfor future use
#Save the model in .h5 format in a folder named saved_models.
model.save('saved_models/model.h5')

# %%
