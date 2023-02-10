import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from AIDetector_pytorch import Detector
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import imutils
from thop import profile
import time
import hopenet, headutils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='weights/hopenet_robust_alpha1.pkl', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video',default='video/simple2.wmv',type=str)      #video/simple.mp4    video/simple2.wmv
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file',default='simple2_result',type=str)
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', default=500,type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    det = Detector()
    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = 'video/head/'
    video_path = args.video_path

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(args.video_path):
        sys.exit('Video does not exist')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # img = torch.zeros((1, 3, 244,244))
    # flops, params = profile(model, inputs=(img,), verbose=False)
    # print(flops, params)

    print('Loading model.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    print('Capture video:{}'.format(video_path))
    cap = cv2.VideoCapture(video_path)

    # New cv2
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    _, im0 = cap.read()
    if _:
        res0 = det.feedCap(im0)
        res0 = res0['frame']
        res0 = imutils.resize(res0, height=500)
        height = res0.shape[0]
        width = res0.shape[1]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('video/%s.mp4' % args.output_string, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    txt_out = open('video/%s.txt' % args.output_string, 'w')

    frame_num,max_frame= 0,1000
    # Start processing frame with bounding box
    # start_time = time.time()
    while True:

        ret, frame = cap.read()
        if ret == False or frame_num>max_frame:
            break
        # cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2_frame,preds= det.detect(frame)
        for pred in preds:
            x_min, y_min, x_max, y_max, conf = int(float(pred[0])), int(float(pred[1])), int(float(pred[2])), int(float(pred[3])), float(pred[5])
            if conf > 0.75:
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                # x_min -= 3 * bbox_width / 4
                # x_max += 3 * bbox_width / 4
                # y_min -= 3 * bbox_height / 4
                # y_max += bbox_height / 4
                x_min -= 50
                x_max += 50
                y_min -= 50
                y_max += 30
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(frame.shape[1], x_max)
                y_max = min(frame.shape[0], y_max)
                # Crop image
                img = cv2_frame[y_min:y_max,x_min:x_max]
                img = Image.fromarray(img)

                # Transform
                img = transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda(gpu)

                yaw, pitch, roll = model(img)

                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)
                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
                print('第{}帧'.format(frame_num),'摇头角度:{}，抬头角度:{}，转头角度:{}'.format(float(yaw_predicted), float(pitch_predicted), float(roll_predicted)))
                # Print new frame with cube and axis
                txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
                headutils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                headutils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                # Plot expanded bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

        frame = imutils.resize(frame, height=500)
        out.write(frame)
        frame_num += 1
        # if frame_num == 1001:
        #     t_delt = time.time() - start_time
        #     print('总用时:',t_delt,'   平均帧率：',1001/t_delt)

    out.release()
    cap.release()
    txt_out.close()
