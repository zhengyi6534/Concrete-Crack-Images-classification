#%%
#1. Import packages
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, metrics, callbacks, applications
from keras import applications


#%%
#2. Data loading 
data_dir = r"C:\Users\User\Documents\PROGRAM_STEP_A101\assessment\Concrete Crack Images for Classification"
BATCH_SIZE = 32
IMG_SIZE = (160,160)
SEED = 42
train_dataset=keras.utils.image_dataset_from_directory(data_dir,validation_split=0.3,subset='training',seed=SEED,batch_size=BATCH_SIZE,image_size=IMG_SIZE,shuffle=True)
val_dataset = keras.utils.image_dataset_from_directory(data_dir,validation_split=0.3,subset='validation',seed=SEED,batch_size=BATCH_SIZE, image_size=IMG_SIZE,shuffle=True)
class_names = train_dataset.class_names


#%%
#3. Convert BatchDataset into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = val_dataset.prefetch(buffer_size=AUTOTUNE)

#%%
#4.Create a small pipeline for data augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#%% 
# Apply the data augmentation to test it out
for images, labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')

#%%
#5. Apply transfer learning
#(A) Import MobileNetV3Large
IMG_SHAPE = IMG_SIZE + (3,)
feature_extractor = keras.applications.MobileNetV3Large(
    input_shape = IMG_SHAPE,include_top=False,
    weights='imagenet',pooling='avg')
feature_extractor.summary()
keras.utils.plot_model(feature_extractor,show_shapes=True)

#%% 
#6. Define the classification layer
l2 = keras.regularizers.L2()

output_layer = layers.Dense(len(class_names),
activation='softmax', kernel_regularizer=l2)

#%%
#7. Use functional API to create the entire model pipeline
inputs = keras.Input(shape=IMG_SHAPE)

x=data_augmentation(inputs)
x=feature_extractor(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

#%%
#8. Compile the model
optimizer=optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=
['accuracy'])

#%%
#9. Evaluate before model training
loss0,acc0 =model.evaluate(pf_test)
print('loss=', loss0)
print('Accuracy=', acc0)

#%%
#10. Create the tensorboard callback
import datetime
log_path = os.path.join('log_dir', 'a3', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_dir=log_path)

#%%
#11. Model training
EPOCHS = 10
history = model.fit(pf_train, validation_data=pf_test,
epochs=EPOCHS,callbacks=[tb])

# %%
#12. Apply model fine tuning strategy
feature_extractor.trainable = True
for layer in feature_extractor.layers[:132]:
    layer.trainable=False

model.summary()

#13. Model compile
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])


# %%
#14. Evaluate the model after training
test_loss, test_acc = model.evaluate(pf_test)
print('Loss=', test_loss)
print('Accuracy=', test_acc)

#%%
#15. Model deployment
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch),axis=1)

plt.figure(figsize=(20,20))

for i in range(len(image_batch)):
    plt.subplot(8,4,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    plt.title(f'Label:{class_names[label_batch[i]]},Prediction: {class_names[y_pred[i]]}')
    plt.axis('off')
plt.show()

#%% Model Saving

# to save trained model
model.save('model.h5')
# %%
