import argparse
import os
import sys
import urllib
import urllib.request
import time
import traceback
import numpy as np
import cv2
from rknn.api import RKNN
from scipy.special import softmax

if __name__ == '__main__':
    # Create RKNN object
    rknn = RKNN(verbose=True)

    #Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[28]], std_values=[[28]],target_platform='rv1106')
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model='mnist.onnx')

    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./data.txt',rknn_batch_size=1)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn('mnist.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')


    # Set inputs
    img = cv2.imread('2.png')
    img = cv2.resize(img, (28, 28))
    #cv2.imwrite('2r.jpg', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 3)

    # Init runtime environment
    print('--> Init runtime environment')

    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])

    #Post Process
    print('--> PostProcess')
    with open('./synset.txt', 'r') as f:
        labels = [l.rstrip() for l in f]

    scores = softmax(outputs[0])
    # print the top-5 inferences class
    scores = np.squeeze(scores)
    a = np.argsort(scores)[::-1]
    print('-----TOP 5-----')
    for i in a[0:5]:
        print('[%d] score=%.2f class="%s"' % (i, scores[i], labels[i]))
    print('done')

    # Release
    rknn.release()
