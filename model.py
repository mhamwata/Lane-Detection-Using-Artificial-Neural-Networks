from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import concatenate, Input, Cropping2D, BatchNormalization, Activation
from keras.optimizers import RMSprop
from keras import backend as k

class unet():
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def crop_layers(self,target,ref):

        target_shape = k.int_shape(target)
        ref_shape = k.int_shape(ref)

        crop_width = (target_shape[2]-ref_shape[2])
        crop_height = (target_shape[1]-ref_shape[1])

        assert (crop_width>=0 & crop_height>=0)

        if crop_width % 2 == 0:
            left_crop, right_crop = int(crop_width/2),int(crop_width/2)
        else:
             left_crop, right_crop = int(crop_width/2),int(crop_width/2)+1
        if crop_height % 2 == 0:
            top_crop, bottom_crop = int(crop_height/2),int(crop_height/2)
        else:
             top_crop, bottom_crop = int(crop_width/2),int(crop_width/2)+1

        return Cropping2D(cropping =((top_crop, bottom_crop),(left_crop, right_crop)))

    def conv_layer(self,filters,kernel_size=3,activation=None):
        return Conv2D(filters=filters, kernel_size=kernel_size, activation=activation,padding='same')

    def deconv_layer(self,filters,kernel_size=2,):
        return Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2,padding = 'same')

    def model(self):
   
        #Encoder
        input = Input(self.input_shape)
        conv1_1 = self.conv_layer(32)(input)
        conv1_1 = BatchNormalization()(conv1_1)
        conv1_1 = Activation('elu')(conv1_1)
        conv1_2 = self.conv_layer(32)(conv1_1)
        conv1_2 = BatchNormalization()(conv1_2)
        conv1_2 = Activation('elu')(conv1_2)
        pool1 = MaxPooling2D(pool_size=2)(conv1_2)
    
        conv2_1 = self.conv_layer(64)(pool1)
        conv2_1 = BatchNormalization()(conv2_1)
        conv2_1 = Activation('elu')(conv2_1)
        conv2_2 = self.conv_layer(64)(conv2_1)
        conv2_2 = BatchNormalization()(conv2_2)
        conv2_2 = Activation('elu')(conv2_2)
        pool2 = MaxPooling2D(pool_size=2)(conv2_2)
    
        conv3_1 = self.conv_layer(128)(pool2)
        conv3_1 = BatchNormalization()(conv3_1)
        conv3_1 = Activation('elu')(conv3_1)
        conv3_2 = self.conv_layer(128)(conv3_1)
        conv3_2 = BatchNormalization()(conv3_2)
        conv3_2 = Activation('elu')(conv3_2)
        pool3 = MaxPooling2D(pool_size=2)(conv3_2)
    
        conv4_1 = self.conv_layer(256)(pool3)
        conv4_1 = BatchNormalization()(conv4_1)
        conv4_1 = Activation('elu')(conv4_1)
        conv4_2 = self.conv_layer(256)(conv4_1)
        conv4_2 = BatchNormalization()(conv4_2)
        conv4_2 = Activation('elu')(conv4_2)
        pool4 = MaxPooling2D(pool_size=2)(conv4_2)
    
        conv5_1 = self.conv_layer(512)(pool4)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = Activation('elu')(conv5_1)
        conv5_2 = self.conv_layer(512)(conv5_1)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_2 = Activation('elu')(conv5_2)
        print(conv5_2.get_shape())
    
        #Concatenated Decoder
        deconv1 = self.deconv_layer(256)(conv5_2)
        print(deconv1.get_shape())
        crop_conv4 = self.crop_layers(conv4_2, deconv1)(conv4_2)
        conc1 = concatenate([crop_conv4,deconv1],axis=3)
        conv6_1 = self.conv_layer(256)(conc1)
        conv6_1 = BatchNormalization()(conv6_1)
        conv6_1 = Activation('elu')(conv6_1)
        conv6_2 = self.conv_layer(256)(conv6_1)
        conv6_2 = BatchNormalization()(conv6_2)
        conv6_2 = Activation('elu')(conv6_2)
    
        deconv2 = self.deconv_layer(128)(conv6_2)
        crop_conv3 = self.crop_layers(conv3_2, deconv2)(conv3_2)
        conc2 = concatenate([crop_conv3,deconv2],axis=3)
        conv7_1 = self.conv_layer(128)(conc2)
        conv7_1 = BatchNormalization()(conv7_1)
        conv7_1 = Activation('elu')(conv7_1)
        conv7_2 = self.conv_layer(128)(conv7_1)
        conv7_2 = BatchNormalization()(conv7_2)
        conv7_2 = Activation('elu')(conv7_2)
    
        deconv3 = self.deconv_layer(64)(conv7_2)
        crop_conv2 = self.crop_layers(conv2_2, deconv3)(conv2_2)
        conc3 = concatenate([crop_conv2,deconv3],axis=3)
        conv8_1 = self.conv_layer(64)(conc3)
        conv8_1 = BatchNormalization()(conv8_1)
        conv8_1 = Activation('elu')(conv8_1)
        conv8_2 = self.conv_layer(64)(conv8_1)
        conv8_2 = BatchNormalization()(conv8_2)
        conv8_2 = Activation('elu')(conv8_2)
    
        deconv4 = self.deconv_layer(32)(conv8_2)
        crop_conv1 = self.crop_layers(conv1_2, deconv4)(conv1_2)
        conc4 = concatenate([crop_conv1,deconv4],axis=3)
        conv9_1 = self.conv_layer(32)(conc4)
        conv9_1 = BatchNormalization()(conv9_1)
        conv9_1 = Activation('elu')(conv9_1)
        conv9_2 = self.conv_layer(32)(conv9_1)
        conv9_2 = BatchNormalization()(conv9_2)
        conv9_2 = Activation('elu')(conv9_2)
    
        conv10 = (self.conv_layer(filters=1, kernel_size=1, activation='sigmoid'))(conv9_2)
        print(conv10._keras_shape)
        model = Model(input = input, output = conv10)
        model.compile(optimizer = RMSprop(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        model.summary()
    
        return model
