###### Importing Libraries ######
import tensorflow as tf
import os
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
#Libaries for dataset spliting
import os
import shutil 
import glob
import math
#Libaries for creating autoencoder and classifier
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, concatenate, BatchNormalization, Average, Dropout
from tensorflow.keras import regularizers


#Finding out how many images the each folder contains under the dataset
root_dir = '/kaggle/input/'  #main folder that contains all images together
number_of_images = {} #creating a dictionary
for dir in os.listdir(root_dir):
  number_of_images[dir] = len(os.listdir(os.path.join(root_dir, dir)))
number_of_images.items()

#making a function to split training and test set images
def train_test_split(folder, split_percentage):
  if not os.path.exists('./'+folder):  #it means if train folder doesn't exist then do the following work
    os.mkdir('./'+folder) #creating a folder named 'train'
    #puting images into the train folder
    for dir in os.listdir(root_dir):
      os.makedirs('./'+folder+'/'+dir) #creatings folder same as under the main folder under train folder as root_dir has
      for img in np.random.choice(a = os.listdir(os.path.join(root_dir, dir)),   #random is used to select random image in folders under train folder
                                size = (math.floor(split_percentage*number_of_images[dir])-8),   #floor is used to give a integer value after all calculation
                                replace = False):
        origin = os.path.join(root_dir, dir, img) #is indicating the source path
        destination = os.path.join('./'+folder, dir)
        shutil.copy(origin, destination)  #this will copy the files from origin to destination folder
        #os.remove(origin)
  else:
    print('The folder already exists...')
    
    
########################### Creating training, test, and validation set ####################################
training_set = train_test_split('train', 0.7)
test_set = train_test_split('test', 0.15)
validation_set = train_test_split('validation', 0.15)
# Path for train, validation and test datasets
train_path = '/kaggle/working/train'
valid_path = '/kaggle/working/validation'
test_path = '/kaggle/working/test'


# Define the path for the empty folder
folder_path = '/kaggle/working/empty_folderf2/'

# Create the empty folder for saving images after epoch end
os.makedirs(folder_path, exist_ok=True)
save_dir = Path('/kaggle/working/empty_folderf2')



####################################### Data Preprocessing ########################################################
train_set = keras.preprocessing.image_dataset_from_directory(
    directory=train_path, label_mode='categorical', image_size=(64,64), batch_size=128,
    shuffle=True
).map(lambda x, y: (x/255.0, y))  # Normalize the pixel values to [0, 1]
 
test_set = keras.preprocessing.image_dataset_from_directory(
    directory=test_path, label_mode='categorical', image_size=(64,64), batch_size=128,
    shuffle=True
).map(lambda x, y: (x/255.0, y))

validation_set =  keras.preprocessing.image_dataset_from_directory(
    directory=valid_path, label_mode='categorical', image_size=(64,64), batch_size=128,
    shuffle=True
).map(lambda x, y: (x/255.0, y))



################################## Creating AUTOENCODER ##########################################

##### Encoder network
input_encoder=(64,64,3)

inputs = keras.Input(shape=input_encoder, name='input_layer')
# Block 1
x = layers.Conv2D(32, kernel_size=3, strides= 1, padding='same', name='conv_1')(inputs)
x = layers.BatchNormalization(name='bn_1')(x)
x = layers.LeakyReLU(name='lrelu_1')(x)

# Block 2
x = layers.Conv2D(64, kernel_size=3, strides= 2, padding='same', name='conv_2')(x)
x = layers.BatchNormalization(name='bn_2')(x)
x = layers.LeakyReLU(name='lrelu_2')(x)

# Block 3
x = layers.Conv2D(64, 3, 2, padding='same', name='conv_3')(x)
x = layers.BatchNormalization(name='bn_3')(x)
x = layers.LeakyReLU(name='lrelu_3')(x)

# Block 4
x = layers.Conv2D(64, 3, 1, padding='same', name='conv_4')(x)
x = layers.BatchNormalization(name='bn_4')(x)
x = layers.LeakyReLU(name='lrelu_4')(x)

# Final Block
flatten = layers.Flatten()(x)
bottleneck = layers.Dense(200, name='dense_1')(flatten)

# Decoder Net
x = layers.Dense(4096, name='dense_reshape')(bottleneck)
x = layers.Reshape((8, 8, 64), name='Reshape_Layer')(x)  # Reshaping to 8x8x64

# Block 1
x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', name='conv_transpose_1')(x)  # Stride 2 to double the size
x = layers.BatchNormalization(name='bnd_1')(x)
x = layers.LeakyReLU(name='lrelud_1')(x)

# Block 2
x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', name='conv_transpose_2')(x)  # Stride 2 to double the size
x = layers.BatchNormalization(name='bnd_2')(x)
x = layers.LeakyReLU(name='lrelud_2')(x)

# Block 3
x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', name='conv_transpose_3')(x)  # Stride 2 to double the size
x = layers.BatchNormalization(name='bnd_3')(x)
x = layers.LeakyReLU(name='lrelud_3')(x)

# Block 4
outputs = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', name='conv_transpose_4')(x)
autoencoder = keras.Model(inputs, outputs)
autoencoder.summary()


######################################## Creating CLASSIFIER ############################################################

# Define input shape
input_shape = (64, 64, 3)

# Define input layer
inputs = Input(shape=input_shape)

# Define path1
path1 = Conv2D(filters=96, kernel_size=(7, 7), activation='relu', strides=(2, 2), padding = 'same')(inputs)
path1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(path1)
path1 = Conv2D(filters=256, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding='same')(path1)
path1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(path1)
path1 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(path1)
path1 = Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(path1)
path1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(path1)
path1 = Flatten()(path1)

path1 = Dropout(0.5)(path1)
path1 = Dense(1024, activation='relu')(path1)
path1 = Dropout(0.5)(path1)
path1 = Dense(1024, activation='relu')(path1)
path1 = Dropout(0.5)(path1)
path1 = Dense(24, activation='softmax')(path1)

outputs = path1

# Create Model
discriminator = Model(inputs=inputs, outputs=outputs)
discriminator.summary()


###################################### Training Phase ############################################

from tensorflow.keras.optimizers import Adam
# Loss function: Mean Squared Error (MSE) between input and output
loss_fn_auto = tf.keras.losses.MeanSquaredError()
loss_fn_disc= tf.keras.losses.CategoricalCrossentropy()
# Optimizer
opt_autoencoder = Adam(learning_rate=0.0005, beta_1=0.5)
opt_disc = Adam(learning_rate=0.00009, beta_1=0.9, beta_2=0.999)
# Compile the autoencoder model with the optimizer and loss function
autoencoder.compile(optimizer=opt_autoencoder, loss=loss_fn_auto)
discriminator.compile(optimizer=opt_disc, loss=loss_fn_disc, metrics=['accuracy'])

train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()

autoencoder_losses_epoch=[]
autoencoder_losses_epoch_val=[]
discriminator_losses_epoch=[]
validation_discriminator_losses_epoch = []
generator_losses = []
discriminator_losses = []
autoencoder_losses = []
autoencoder_losses_val = []
val_dsc_losses = []
train_acc = []
val_acc = []
train_acc_epoch = []
val_acc_epoch = []
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import numpy as np

epochs = 50

# Training loop
for epoch in range(epochs):

    for idx, (batch_real, batch_label) in enumerate(tqdm(train_set)):
        batch_size = batch_real.shape[0]
        
        #random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        #print("real: ", batch_label)

        # Autoencoder (Generator) Training
        with tf.GradientTape() as autoencoder_tape:
            generated_image= autoencoder(batch_real, training = True)
            loss_autoencoder = loss_fn_auto(batch_real, generated_image)

        grads_autoencoder = autoencoder_tape.gradient(loss_autoencoder, autoencoder.trainable_weights)
        opt_autoencoder.apply_gradients(zip(grads_autoencoder, autoencoder.trainable_weights))

        # Get the generated image from the autoencoder (generator)
        fake = generated_image
        
        
        # Discriminator Training
        with tf.GradientTape() as disc_tape:
            y_hat = discriminator(batch_real)
            loss_disc_real = loss_fn_disc(batch_label, y_hat)
            #print("y_hat: ",y_hat)
            loss_disc_fake = loss_fn_disc(batch_label, discriminator(fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

        grads_discriminator = disc_tape.gradient(loss_disc, discriminator.trainable_weights)
        opt_disc.apply_gradients(zip(grads_discriminator, discriminator.trainable_weights))
        
        # Update training metric.
        train_acc_metric.update_state(batch_label, y_hat)
        
        autoencoder_losses.append(loss_autoencoder.numpy())
        discriminator_losses.append(loss_disc.numpy())

        # Save generated images at intervals
        if idx % 100 == 0:
            img = keras.preprocessing.image.array_to_img(fake[0])
            img.save(f"{save_dir}/generated_img_epoch_{epoch}_batch_{idx}.png")
            
    # Training accuracy
    train_acc = train_acc_metric.result()

    
            
    for x_batch_val, y_batch_val in validation_set:
        val_generated_images = autoencoder(x_batch_val, training = False)
        val_loss_auto = loss_fn_auto(x_batch_val, val_generated_images)
        autoencoder_losses_val.append(val_loss_auto)
        
        val_logits = discriminator(x_batch_val, training=False)
        val_loss_disc_real = loss_fn_disc(y_batch_val, val_logits)
        val_dsc_losses.append(val_loss_disc_real)
        val_acc_metric.update_state(y_batch_val, val_logits)  
        
    #Validation Accuracy    
    val_acc = val_acc_metric.result()
    avg_val_dsc_loss = np.mean(val_dsc_losses)
    
    #saving all values in an array
    avg_autoencoder_loss = np.mean(autoencoder_losses)
    avg_discriminator_loss = np.mean(discriminator_losses)
    # Append epoch-wise losses to the lists
    autoencoder_losses_epoch.append(avg_autoencoder_loss)
    discriminator_losses_epoch.append(avg_discriminator_loss)
    validation_discriminator_losses_epoch.append(avg_val_dsc_loss)
    autoencoder_losses_epoch_val.append(np.mean(autoencoder_losses_val))
    train_acc_epoch.append(train_acc)
    val_acc_epoch.append(val_acc)
    #print(len(val_acc_epoch))

    #Display values
    print(f"Epoch {epoch+1}/{epochs}: Autoencoder Loss: {avg_autoencoder_loss}, Training Classifer Loss: {avg_discriminator_loss}, Validation Classifier Loss: {avg_val_dsc_loss}")
    print(f'Training Accuracy: {train_acc}, Validation Accuracy: {val_acc}')
    if epoch == 0:
        print(f"The model is saving from -inf to {val_acc}")
        best_acc = val_acc
    elif epoch > 0 and val_acc> best_acc:
        best_acc = val_acc
        discriminator.save('/kaggle/working/my_model.keras')
        print(f"Saving the model accuracy to {best_acc}")
    else: 
        print(f'The accuracy did not improve from {best_acc}')
        
    #Reset training metrics at the end of each epoch  
    val_acc_metric.reset_states()
    train_acc_metric.reset_states()


############################ Evaluation ##########################################
new_model = tf.keras.models.load_model('my_model.keras')
new_model.evaluate(test_set)
new_model.evaluate(train_set)
new_model.evaluate(validation_set)

import numpy as np

# Initialize variables to store TP, FP, FN, precision, recall, and F1 score for each class
true_positives = np.zeros(24)
false_positives = np.zeros(24)
false_negatives = np.zeros(24)
precision = np.zeros(24)
recall = np.zeros(24)
f1_score = np.zeros(24)

# Loop through the validation set to compute TP, FP, and FN for each class
for x_batch_val, y_batch_val in test_set:
    # Get predicted labels from the discriminator
    val_logits = new_model(x_batch_val, training=False)
    predicted_labels = np.argmax(val_logits, axis=1)
    
    # Compare predicted labels with true labels to compute TP, FP, and FN
    for i in range(len(y_batch_val)):
        true_label = np.argmax(y_batch_val[i])
        predicted_label = predicted_labels[i]
        if predicted_label == true_label:
            true_positives[true_label] += 1
        else:
            false_positives[predicted_label] += 1
            false_negatives[true_label] += 1

# Calculate precision, recall, and F1 score for each class
for i in range(24):
    if true_positives[i] + false_positives[i] == 0:
        precision[i] = 0
    else:
        precision[i] = true_positives[i] / (true_positives[i] + false_positives[i])
    
    if true_positives[i] + false_negatives[i] == 0:
        recall[i] = 0
    else:
        recall[i] = true_positives[i] / (true_positives[i] + false_negatives[i])
    
    if precision[i] + recall[i] == 0:
        f1_score[i] = 0
    else:
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

# Average precision, recall, and F1 score across all classes
average_precision = np.mean(precision)
average_recall = np.mean(recall)
average_f1_score = np.mean(f1_score)


print("Average Precision:", average_precision)
print("Average Recall:", average_recall)
print("Average F1 Score:", average_f1_score)


################################### Plots ###########################################

#Loss Plot
import matplotlib.pyplot as plt
plt.suptitle('Loss Plot', fontsize=14, fontweight='bold')
plt.plot(discriminator_losses_epoch, 'r', label='Train Loss')
plt.plot(validation_discriminator_losses_epoch, 'g', label='Validation Classifier Loss')
plt.legend()
plt.xlabel('Epochs', fontsize=12, fontstyle='normal')
plt.ylabel('Loss', fontsize=12, fontstyle='normal')
#plt.grid(color='black', linestyle='-', linewidth=1)
plt.savefig('/kaggle/working/classifier_loss_plot.png')
plt.show()

#Accuracy Plot
plt.suptitle('Accuracy Plot', fontsize=14, fontweight='bold')
plt.plot(train_acc_epoch, 'r', label='Train Accuracy')
plt.plot(val_acc_epoch, 'g', label='Validation Accuracy')
plt.xlabel('Epochs', fontsize=12, fontstyle='normal')
plt.ylabel('Accuracy', fontsize=12, fontstyle='normal')
plt.legend()
plt.savefig('/kaggle/working/accuracy_plot.png')
plt.show()

#Autoencoder Loss Plot
plt.suptitle('Autoencoder Loss Plot', fontsize=14, fontweight='bold')
plt.plot(autoencoder_losses_epoch, 'r', label='Train Loss')
plt.plot(autoencoder_losses_epoch_val, 'g', label='Validation Loss')
plt.legend()
plt.xlabel('Epochs', fontsize=12, fontstyle='normal')
plt.ylabel('Loss', fontsize=12, fontstyle='normal')
#plt.grid(color='black', linestyle='-', linewidth=1)
plt.savefig('/kaggle/working/loss_plot.png')
plt.show()
