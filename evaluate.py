import glob
import os
from load_data import *
from tensorboard import *
from model import *
from train import *
from keras.callbacks import TensorBoard, ModelCheckpoint


#Define directories for training and test images
train_dir = 'data_road\\train'
test_dir = 'data_road\\test'

#Hyperparameters for Training
EPOCHS = 20
VAL_SPLIT = 0.2
BATCH_SIZE = 30
VERBOSE = 1
TRAIN = 1
steps_per_epoch = 10

#Image shape
HEIGHT = 240
WIDTH = 432
CHANNELS = 3
input_shape = (HEIGHT,WIDTH,CHANNELS)

#Visualize training with Tensorboard
tensorboard_scalars = TrainValTensorBoard(write_graph=False,histogram_freq=1)

#Save weights at checkpoints
checkpoint = ModelCheckpoint('best_model.h5',monitor='acc',verbose=VERBOSE,save_best_only=True)

#Callbacks in single list
callbacks_list = [tensorboard_scalars,checkpoint]

#load data
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=10,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     validation_split=VAL_SPLIT)
seed = 1

#load data
model_generator = load(train_dir,data_gen_args,BATCH_SIZE,(HEIGHT,WIDTH),seed,TRAIN)


#Train
network = trainer(input_shape)
model = network.model_load()

model.fit_generator(model_generator,steps_per_epoch=steps_per_epoch,epochs=EPOCHS,callbacks=callbacks_list)
model.save('unet_model.h5')
model.save_weights('unet_weights.h5')


#img = cv2.imread('C:/Users/Mwinde/OneDrive/Lane Detection/Youtube Video Videocaps/test_images/frame6s\-.jpg')
#img = resize_img(img,(WIDTH,HEIGHT))
#testor = np.expand_dims(img,axis=0)
#predictions = model.predict(test_batch,1,VERBOSE)
#print(predictions.shape)

#for i in range(0,predictions.shape[0]):
#    act_map = predictions[i]
#    act_map /= np.amax(act_map)
#    act_map *= 255
#    act_map = act_map.astype(np.uint8)
#    act_map = np.expand_dims(cv2.resize(act_map,init_shape),axis=2)
#    print('Activation Map: ',act_map.shape)
#    lane_up = np.array([255])
#    lane_down = np.array([250])
#    mask = cv2.inRange(act_map, lane_down, lane_up)
#    green = np.zeros((init_img_shape[1],init_img_shape[0],3),dtype=np.uint8)
#    green[:,:,1] = 255
#    print('Green: ',green.shape)
#    print('Mask: ',mask.shape)
#    road_mask = cv2.bitwise_and(green,green,mask=np.expand_dims(mask,axis=2))
#    test_batch[i] = cv2.resize(test_batch[i],(init_shape[1],init_shape[0]))
#    img_with_mask = cv2.addWeighted(src1= test_batch[i],axis=2,alpha=1,src2=road_mask,beta=1,gamma=0)
#    #Save result
#    cv2.imwrite('Youtube Video Videocaps/Activation Map {}.png'.format(i),act_map)
#    cv2.imwrite('Youtube Video Videocaps/Mask {}.png'.format(i),road_mask)
#    cv2.imwrite('Youtube Video Videocaps/Prediction {}.png'.format(i),img_with_mask)


#j = 0
#for i in range(0,test_batch.shape[0]):
#    network.test_model(test_batch[i],j)
#    print('Done with {}'.format(j))
#    j += 1

