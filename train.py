from model import unet
from keras.models import load_model
from keras import backend as K
import os
import numpy as np
import cv2

class trainer():
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def model_load(self,model_h5=None):
        if model_h5 == None:
            network = unet(self.input_shape)
            model = network.model()
            self.model = model
            return model
        else:
            model = load_model(model_h5)
            self.model = model
            print('Model loaded from file {}'.format(model_h5))
            return model

    def train_model(self,x,y,batch_size,epochs,verbose,validation_split,callbacks,augment=False):
        model = self.model
        if augment==False:
            model.fit(x,y, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split, callbacks=callbacks)
        else:
            model.fit_generator()
        model.save('unet_model.h5')
    
    def test_model(self,img,index):
        img = resize_img(img,(self.input_shape[1],self.input_shape[0]))
        input = self.model.input
        outputs = [layer.output for layer in self.model.layers]
        func = K.function([input,K.learning_phase()],outputs)
        layer_outputs = func([np.expand_dims(img,axis=0),1])
        act_map = layer_outputs[0][-1]
        act_map /= np.amax(act_map)
        act_map *= 255
        act_map = act_map.astype(np.uint8)
        lane_up = np.array([255])
        lane_down = np.array([250])
        mask = cv2.inRange(act_map, lane_down, lane_up)
        green = np.zeros(self.input_shape,dtype=np.uint8)
        green[:,:,1] = 255
        road_mask = cv2.bitwise_and(green,green,mask=mask)
        img_with_mask = cv2.addWeighted(src1= img,alpha=1,src2=road_mask,beta=1,gamma=0)
        #Save result
        os.makedir('results')
        cv2.imwrite('results/Activation Map {}.png'.format(index),act_map)
        cv2.imwrite('results/Mask {}.png'.format(index),road_mask)
        cv2.imwrite('results/Prediction {}.png'.format(index),img_with_mask)


