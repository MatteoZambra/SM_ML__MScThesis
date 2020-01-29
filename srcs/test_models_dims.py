

from keras.models import load_model


model_init_mnist = load_model(r'C:\users\matte\_nmnn\srcs\Results\seed_618\mnist\orth\init\model_init.h5')
model_init_synds = load_model(r'C:\users\matte\_nmnn\srcs\Results\seed_618\synds\orth\init\model_init.h5')

model_init_mnist.summary()
model_init_synds.summary()