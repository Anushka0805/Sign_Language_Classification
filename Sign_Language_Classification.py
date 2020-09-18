
from sign_language import *

#provide the path to Gesture Image Pre-Processed Data
inpath='/.../Gesture Image Pre-Processed Data/'
#provide the path to the folder to store all images in
outpath='/.../dataset'

X_train,X_test,X_val,Y_train,Y_test,Y_val = split_data(inpath, outpath)

model = load_model('SL_model.h5')
results = model.evaluate(X_test, Y_test, batch_size=128)
print("test loss, test acc:", results)

#provide the path to the input image
path='/.../eg1.jpg'
predict_img_data(path, model)
