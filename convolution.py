import numpy
from scipy import signal

class Convolution:

    def convolve(self,input,filters,conv_shape):

        n_input = input.shape[0]
        n_featuremaps = input.shape[1]
        input_height = input.shape[2]  # image dimension
        input_width = input.shape[3]
        #Input Shape : ( 400 X 64 X 112 X 92 )

        n_channel_in = filters.shape[0] # same as n_featuremaps
        n_channel_out = filters.shape[1] # same as n_filters
        filter_h = filters.shape[2]
        filter_w = filters.shape[3]
        #Filter Shape : ( 64 X 16 X 3 X 3 )

        conv_height = conv_shape[2]  # size of convolved image
        conv_width = conv_shape[3]

        convolved_features = numpy.zeros(conv_shape)  # total number of convolved features

        for input_num in range(n_input):
            for channel_out in range(n_channel_out): # for each filter
                """ Image to be convolved """
                convolved_image = numpy.zeros((conv_height, conv_width))  # create an zero initialized convolved image
                for channel_in in range(n_channel_in): # for each feature map

                    filter = filters[channel_in,channel_out,:,:]
                    image = input[input_num, channel_in, :, :]
                    """ Convolve image with the feature and add to existing matrix """
                    convolved_image = convolved_image + signal.convolve2d(image, filter,'same')

                convolved_features[input_num,channel_out,:,:] = convolved_image
        return convolved_features



    def convolve_backprop(self, input, convout_grad, filters, filters_grad):
        """
        Back-propagate gradients of multi-image, multi-channel convolution
        imgs has shape (n_imgs, n_channels_in, img_h, img_w)
        filters has shape (n_channels_in, n_channels_out, img_h, img_w)
        convout has shape (n_imgs, n_channels_out, img_h, img_w)
        """

        n_imgs = convout_grad.shape[0]
        img_h = convout_grad.shape[2]
        img_w = convout_grad.shape[3]
        n_channels_convout = filters.shape[1]
        n_channels_imgs = filters.shape[0]
        fil_h = filters.shape[2]
        fil_w = filters.shape[3]
        fil_mid_h = fil_h // 2
        fil_mid_w = fil_w // 2

        imgs_grad = numpy.zeros((n_imgs,n_channels_imgs,img_h,img_w))  # total number of convolved features
        filters_grad[...] = 0
        for i in range(n_imgs):
            for c_convout in range(n_channels_convout):
                for y in range(img_h):
                    y_off_min = max(-y, -fil_mid_h)
                    y_off_max = min(img_h - y, fil_mid_h + 1)
                    for x in range(img_w):
                        convout_grad_value = convout_grad[i, c_convout, y, x]
                        x_off_min = max(-x, -fil_mid_w)
                        x_off_max = min(img_w - x, fil_mid_w + 1)
                        for y_off in range(y_off_min, y_off_max):
                            for x_off in range(x_off_min, x_off_max):
                                img_y =  (y + y_off)
                                img_x = (x + x_off)
                                fil_y = (fil_mid_w + y_off)
                                fil_x = (fil_mid_h + x_off)
                                for c_imgs in range(n_channels_imgs):
                                    imgs_grad[i, c_imgs, img_y, img_x] += filters[c_imgs, c_convout, fil_y, fil_x] * convout_grad_value
                                    filters_grad[c_imgs, c_convout, fil_y, fil_x] += (input[ i, c_imgs, img_y, img_x] * convout_grad_value)

        filters_grad[...] /= (img_h*img_w)

        return imgs_grad,filters_grad


