from PIL import Image
import numpy as np
from keras import Sequential
from keras.layers import Conv2D

# import first picture and change to an array.
input_image = Image.open('../../res/image/120x80/120x80 (1).png')
input_image_array = np.array(input_image)

# define nearest neighbor interpolation function which can expand origin picture to the destination size.
def nearest_neighbor_interpolation(input_img, destination_height, destination_width):
    source_height, source_width, _ = input_img.shape
    return_image = np.zeros((destination_height, destination_width, 3), dtype=np.uint8)
    for x in range(destination_height):
        for y in range(destination_width):
            source_x = round((x + 1) * (source_height / destination_height))
            source_y = round((y + 1) * (source_width / destination_width))
            return_image[x, y] = input_img[source_x - 1, source_y - 1]
    return return_image

# resize the origin picture to destination size
image_low_resolution = nearest_neighbor_interpolation(input_image_array, input_image_array.shape[0] * 4,
                                                      input_image_array.shape[1] * 4)

# cut the origin picture (which has low resolution with the destination size) to 33*33 px fragments as the input data.
# 0~32, 33~65, 66~98 ... but the last one is 447~479 which has the same part with the 462~459.
input_data = np.zeros((150, 33, 33, 3))
for h in range(0, 10):
    for w in range(0, 15):
        if h < 9:
            if w < 14:
                input_data[h * 10 + w] = image_low_resolution[h * 33:(h + 1) * 33, w * 33:(w + 1) * 33]
            else:
                input_data[h * 10 + w + 1] = image_low_resolution[h * 33:(h + 1) * 33, 480 - 33:480]
        else:
            if w < 14:
                input_data[h * 10 + w + 1] = image_low_resolution[320 - 33:320, w * 33:(w + 1) * 33]
            else:
                input_data[h * 10 + w + 1] = image_low_resolution[320 - 33:320, 480 - 33:480]

# use the same way to cut real picture to 33*33 px fragments as the input labels.
input_label = np.zeros((150, 21, 21, 3))
image_high_resolution = np.array(Image.open('../../res/image/480x320/480x320 (1).png'))
for h in range(0, 10):
    for w in range(0, 15):
        if h < 9:
            if w < 14:
                input_label[h * 10 + w] = image_high_resolution[h * 33 + 6:(h + 1) * 33 - 6,
                                          w * 33 + 6:(w + 1) * 33 - 6]
            else:
                input_label[h * 10 + w + 1] = image_high_resolution[h * 33 + 6:(h + 1) * 33 - 6, 480 - 33 + 6:480 - 6]
        else:
            if w < 14:
                input_label[h * 10 + w + 1] = image_high_resolution[320 - 33 + 6:320 - 6, w * 33 + 6:(w + 1) * 33 - 6]
            else:
                input_label[h * 10 + w + 1] = image_high_resolution[320 - 33 + 6:320 - 6, 480 - 33 + 6:480 - 6]

# create the SRCNN model.
# first layer is 2D convolution with the ReLU activation, the number of the filter is 64, filter size is (9, 9).
# second layer is 2D convolution with the ReLU activation, the number of the filter is 32, filter size is (1, 1).
# third layer is 2D convolution, the number of the filter is 3, filter size is (5, 5).
model = Sequential()
model.add(Conv2D(64, (9, 9), data_format='channels_last', activation='relu', input_shape=(33, 33, 3)))
model.add(Conv2D(32, (1, 1), data_format='channels_last', activation='relu'))
model.add(Conv2D(3, (5, 5), data_format='channels_last', activation='linear'))

# the loss function is MSE, the optimizer is SGD.
model.compile(loss='mean_squared_error', optimizer='sgd')

# input data and labels, just only train one time.
model.fit(input_data, input_label, epochs=1)
print(model.summary())
