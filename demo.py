from AIDetector_pytorch import Detector
import imutils
import cv2

def main():

    name = 'demo'
    det = Detector()
    cap = cv2.VideoCapture('video/simple.mp4')
    # rtsp://dsmy:dsmy123@!@192.168.3.96:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif
    # video/detect_test.mp4   video/cars.mp4     video/realroad.mp4        video/simple.mp4
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('fps:', fps)
    # t = int(1000/fps)
    _,im0=cap.read()
    if _:
        res0 = det.feedCap(im0)
        res0 = res0['frame']
        res0 = imutils.resize(res0, height=500)
        height = res0.shape[0]
        width = res0.shape[1]

    videoWriter = True
    if videoWriter:
        videoWriter = cv2.VideoWriter(
            'video/result0.mp4',  cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (width,height))
    i,max_frame = 0,1500
    while True:
        # try:
        _, im = cap.read()
        if (im is None) or i > max_frame:
            break
        
        result = det.feedCap(im)
        result = result['frame']
        result = imutils.resize(result, height=500)
        videoWriter.write(result)
        i=i+1
        # cv2.imshow(name, result)
        # cv2.waitKey(t)

        # if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
        #     # 点x退出
        #     break
        # except Exception as e:
        #     print(e)
        #     break
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    
    main()