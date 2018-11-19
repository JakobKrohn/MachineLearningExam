import numpy as np
import PIL.Image as pil
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist')

testImage = (np.array(mnist.test.images[0], dtype='float')).reshape(28,28)

img = pil.fromarray(np.uint8(testImage * 255), 'L')
img.show()
