import numpy as np
from PIL import Image
from keras import Sequential
from keras.layers import Conv2D


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


input_x = np.zeros((0, 33, 33, 3))
input_y = np.zeros((0, 21, 21, 3))
for num in range(1, 66):
    # import picture and change to an array.
    image_path = '../../res/image/120x80/120x80 (' + str(num) + ').png'
    input_image = Image.open(image_path)
    input_image_array = np.array(input_image)
    # resize the origin picture to destination size
    image_low_resolution = nearest_neighbor_interpolation(input_image_array, input_image_array.shape[0] * 4,
                                                          input_image_array.shape[1] * 4)
    # cut the origin picture (which has low resolution with the destination size) to 33*33 px fragments
    # as the input data.
    # 0~32, 33~65, 66~98 ... but the last one is 447~479 which has the same part with the 462~459.
    input_data = np.zeros((150, 33, 33, 3))
    for height in range(0, 10):
        for weight in range(0, 15):
            if height < 9:
                if weight < 14:
                    input_data[height * 10 + weight] = image_low_resolution[
                                                       height * 33:(height + 1) * 33,
                                                       weight * 33:(weight + 1) * 33]
                else:
                    input_data[height * 10 + weight + 1] = image_low_resolution[
                                                           height * 33:(height + 1) * 33,
                                                           480 - 33:480]
            else:
                if weight < 14:
                    input_data[height * 10 + weight + 1] = image_low_resolution[
                                                           320 - 33:320,
                                                           weight * 33:(weight + 1) * 33]
                else:
                    input_data[height * 10 + weight + 1] = image_low_resolution[
                                                           320 - 33:320,
                                                           480 - 33:480]
    input_x = np.append(arr=input_x, values=input_data, axis=0)

    # use the same way to cut real picture to 33*33 px fragments as the input labels.
    image_path = '../../res/image/480x320/480x320 (' + str(num) + ').png'
    input_image = Image.open(image_path)
    input_image_array = np.array(input_image)
    image_low_resolution = nearest_neighbor_interpolation(input_image_array, input_image_array.shape[0] * 4,
                                                          input_image_array.shape[1] * 4)
    input_label = np.zeros((150, 21, 21, 3))
    for height in range(0, 10):
        for weight in range(0, 15):
            if height < 9:
                if weight < 14:
                    input_label[height * 10 + weight] = image_low_resolution[
                                                        height * 33 + 6:(height + 1) * 33 - 6,
                                                        weight * 33 + 6:(weight + 1) * 33 - 6]
                else:
                    input_label[height * 10 + weight + 1] = image_low_resolution[
                                                            height * 33 + 6:(height + 1) * 33 - 6,
                                                            480 - 33 + 6:480 - 6]
            else:
                if weight < 14:
                    input_label[height * 10 + weight + 1] = image_low_resolution[
                                                            320 - 33 + 6:320 - 6,
                                                            weight * 33 + 6:(weight + 1) * 33 - 6]
                else:
                    input_label[height * 10 + weight + 1] = image_low_resolution[
                                                            320 - 33 + 6:320 - 6,
                                                            480 - 33 + 6:480 - 6]
    input_y = np.append(arr=input_y, values=input_label, axis=0)
    print('picture ' + str(num) + ' had been loaded!')
print('------loading finish!------')

# create the SRCNN model.
model = Sequential()
# first layer is 2D convolution with the ReLU activation, the number of the filter is 64, filter size is (9, 9).
model.add(Conv2D(64, (9, 9), data_format='channels_last', activation='relu', input_shape=(33, 33, 3)))
# second layer is 2D convolution with the ReLU activation, the number of the filter is 32, filter size is (1, 1).
model.add(Conv2D(32, (1, 1), data_format='channels_last', activation='relu'))
# third layer is 2D convolution, the number of the filter is 3, filter size is (5, 5).
model.add(Conv2D(3, (5, 5), data_format='channels_last', activation='linear'))

# the loss function is MSE, the optimizer is SGD.
model.compile(loss='mean_squared_error', optimizer='sgd')

# input data and labels, just only train one time.
model.fit(x=input_x, y=input_y, batch_size=150, epochs=10)
print(model.summary())
