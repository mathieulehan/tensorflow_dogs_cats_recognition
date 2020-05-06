from keras.models import load_model
import tensorflow as tfjs

# get model
modelFile = 'model_20species75train25test'
model = load_model(modelFile)

# convert model
tfjs.convert_to_tensor(model)