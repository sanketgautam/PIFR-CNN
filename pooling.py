import numpy

class Pooling:

    def pool(self,input,pool_h,pool_w,last_maxpositions,mode='max'):

        n_input = input.shape[0]
        n_featuremaps = input.shape[1]
        input_height = input.shape[2]  # image dimension
        input_width = input.shape[3]

        res_height = input_height // pool_h  # resulting dimension
        res_width = input_width // pool_w

        """ Initialize pooled features as array of zeros """
        pooled_features = numpy.zeros((n_input, n_featuremaps, res_height, res_width))

        for input_num in range(n_input):
            for feature_num in range(n_featuremaps):
                for pool_row in range(res_height):

                    row_start = pool_row * pool_h
                    row_end = row_start + pool_h

                    for pool_col in range(res_width):
                        col_start = pool_col * pool_w
                        col_end = col_start + pool_w

                        """ Extract image patch and calculate max pool """
                        patch = input[ input_num, feature_num,row_start:row_end, col_start:col_end]

                        if mode == 'max':
                            maxpos = numpy.argmax(patch,axis = 1)
                            last_maxpositions[input_num,feature_num,pool_row,pool_col,0]=maxpos[0]+row_start
                            last_maxpositions[input_num, feature_num, pool_row, pool_col,1] = maxpos[1]+col_start
                            pooled_features[input_num,feature_num,pool_row,pool_col] = numpy.max(patch)

        return pooled_features,last_maxpositions



    def pool_backprop(self,output_grad,output_shape,last_maxpositions):

        n_input = output_grad.shape[0]
        n_featuremaps = output_grad.shape[1]
        poolout_h = output_grad.shape[2]
        poolout_w = output_grad.shape[3]

        input_grad = numpy.zeros(output_shape)

        for input_num in range(n_input):
            for feature_num in range(n_featuremaps):
                for poolout_x in range(poolout_h):
                    for poolout_y in range(poolout_w):
                        maxpos_x = last_maxpositions[input_num,feature_num,poolout_x,poolout_y,0]
                        maxpos_y = last_maxpositions[input_num, feature_num, poolout_x, poolout_y,1]
                        input_grad[input_num,feature_num,maxpos_x,maxpos_y]=output_grad[input_num,feature_num,poolout_x,poolout_y]
        return input_grad
