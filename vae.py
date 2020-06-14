import utils
import tflite_runtime.interpreter as tflite
import platform
import sys, os, io
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#import cv2
from PIL import Image
import numpy as np

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    print(model_file)
    return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

def main():
    print("hello world")
    interpreter = make_interpreter("models/quantized_vae.tflite")
    interpreter.allocate_tensors()
    size = utils.input_size(interpreter)
    print(size)
    #image = cv2.imread("0.0722970757_0.223937735123.png", 0)
    #image = cv2.resize(image, (200, 120), interpolation=cv2.INTER_AREA)  # width(x), height(y)
    #with open("test.jpg", "rb") as f:
    #    b = io.BytesIO(f.read())
    image = Image.open('test.jpg')
    image = image.convert('L')#.resize(size, Image.ANTIALIAS)
    image = np.expand_dims(np.array(image).astype(np.float32), axis=2)
    print(image[image > 0])
    utils.set_input(interpreter, image)
    image_x = utils.get_output(interpreter)
    print(image_x[image_x > 0])
    print(image_x.shape)
    image_x = Image.fromarray(image_x.reshape(120,200))
    new_p = image_x.convert("L")
    new_p.save('out.jpg')

if __name__ == "__main__":
    main()
