import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Layer
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.losses import Loss
import keras.backend as K

batch_size=256
EPOCHS=100

train_folder = "./Data/Number Plates/new plates/train"
val_folder = "./Data/Number Plates/new plates/valid"


train_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(
    directory=train_folder,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical'
)


val_datagen = ImageDataGenerator(rescale=1./255.)

val_generator = val_datagen.flow_from_directory(
    directory=val_folder,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical'
)

#Visualize Train Set

# x,y = train_generator.next()
# for i in range(0,10):
#     image= x[i]
#     plt.imshow(image)
#     plt.show()


#Define the VGG-16 Model

class CustomDenseLayer(Layer):
    def __init__(self,units,activation):
        super().__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    def build(self,input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(trainable=True,name="kernel",initial_value=w_init(shape=(input_shape[-1],self.units),dtype="float32"))
        
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(trainable=True,name="bias",initial_value=b_init(shape=(self.units,),dtype="float32"))
         
    def call(self,inputs):
        return self.activation(tf.matmul(inputs,self.w) + self.b)

inputs = Input(shape=(224,224,3),batch_size=batch_size,name='Input Layer',dtype=tf.float32)

x = Conv2D(filters=3,kernel_size=(64,64),strides=(1,1),padding="same",name="Conv1",activation='relu')(inputs)
x = Conv2D(filters=3,kernel_size=(64,64),strides=(1,1),padding="same",name="Conv2",activation='relu')(x)
x = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same",name="Maxpool1")(x)

x = Conv2D(filters=3,kernel_size=(128,128),strides=(1,1),padding="same",name="Conv3",activation='relu')(x)
x = Conv2D(filters=3,kernel_size=(128,128),strides=(1,1),padding="same",name="Conv4",activation='relu')(x)
x = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same",name="Maxpool2")(x)

x = Conv2D(filters=3,kernel_size=(256,256),strides=(1,1),padding="same",name="Conv5",activation='relu')(x)
x = Conv2D(filters=3,kernel_size=(256,256),strides=(1,1),padding="same",name="Conv6",activation='relu')(x)
x = Conv2D(filters=3,kernel_size=(256,256),strides=(1,1),padding="same",name="Conv7",activation='relu')(x)
x = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same",name="Maxpool3")(x)

x = Conv2D(filters=3,kernel_size=(512,512),strides=(1,1),padding="same",name="Conv8",activation='relu')(x)
x = Conv2D(filters=3,kernel_size=(512,512),strides=(1,1),padding="same",name="Conv9",activation='relu')(x)
x = Conv2D(filters=3,kernel_size=(512,512),strides=(1,1),padding="same",name="Conv10",activation='relu')(x)
x = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same",name="Maxpool4")(x)

x = Conv2D(filters=3,kernel_size=(512,512),strides=(1,1),padding="same",name="Conv11",activation='relu')(x)
x = Conv2D(filters=3,kernel_size=(512,512),strides=(1,1),padding="same",name="Conv12",activation='relu')(x)
x = Conv2D(filters=3,kernel_size=(512,512),strides=(1,1),padding="same",name="Conv13",activation='relu')(x)
x = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same",name="Maxpool5")(x)


x = Flatten(name="flatten")(x)
x = CustomDenseLayer(units=4096,activation='relu')(x)
x = CustomDenseLayer(units=4096,activation='relu')(x)
x = CustomDenseLayer(units=56,activation='relu')(x)
outputs = tf.keras.layers.Softmax(axis=-1)(x)


class Categorical__Crossentropy(Loss):
    def __init__(self):
        super().__init__()

    def call(self,y_true,y_pred):

        epsilon  = 1e-15
        y_pred = tf.maximum(y_pred,epsilon)
        loss = -K.sum(y_true * K.log(y_pred)) 
        
        return loss



model = tf.keras.Model(inputs=inputs,outputs=outputs)
# tf.keras.utils.plot_model(model,to_file="model.png",show_shapes=True,show_layer_activations=True)
model.compile(optimizer='sgd',loss=Categorical__Crossentropy(),metrics=['accuracy'])
model.fit(train_generator,batch_size=batch_size,epochs=EPOCHS,validation_data=(val_generator),validation_batch_size=batch_size)


