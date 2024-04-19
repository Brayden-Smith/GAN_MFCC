import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import display
from tensorflow.keras import layers
import time
import soundfile as sf
import librosa

#makes computation lesser so that it completes in time probably annihlates accuracy
BATCH_SIZE = 1
    
#idk why this makes it work but it does
tf.config.run_functions_eagerly(True)

def make_generator_model():
    model = tf.keras.Sequential()


    model.add(layers.Dense(5 * 5 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())


    model.add(layers.Reshape((5, 5, 256)))
    assert model.output_shape == (None, 5, 5, 256)


    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 5, 5, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())


    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 10, 10, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())


    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 20, 20, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Resizing(20, 5168, interpolation='bilinear'))

    # Final layer to adjust depth to 1
    model.add(layers.Conv2D(1, (3, 3), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 20, 5168, 1)

    return model



generator = make_generator_model()




noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[20, 5168, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#saves weights so that it isnt tossed out
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])


def train_step(images):
    print("Starting train_step")
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    print("Noise generated")

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        print("Generated images")

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        print("Discriminator output calculated")

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        print("Losses calculated")

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    print("Gradients calculated")

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
def train(path, epochs):
  for epoch in range(epochs):
    start = time.time()

    for filename in os.listdir(path):
        image_batch = np.load(path + "/" + filename)

        image_batch = image_batch.reshape(1, 20, 5168, 1)

        train_step(image_batch)



    # Save the model every epoch
    checkpoint.save(file_prefix = checkpoint_prefix)


print(generator.summary())

train("normData/normTrainData", EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

noise_dim = 100
num_examples_to_generate = 10
seed = tf.random.normal([num_examples_to_generate, noise_dim])
examples = generator(seed,training=False)

# gotta unnormalize each array save it then make a wav version
for example in examples:
    std = np.load("TrainingStdImportant.npy")
    example = np.squeeze(example,axis=2)
    example = np.multiply(example,std)

    mean = np.load("TrainingMeanImportant.npy")
    example = np.add(example,mean)

    np.save('output/' + str(example[0][0]),example)

    #just gonna give it the first index name cause its easy to not have it overwritefdd
    audio = librosa.feature.inverse.mfcc_to_audio(example)
    sf.write('output/' + str(example[0][0]) + '.wav', audio, 44100, subtype='PCM_24')

