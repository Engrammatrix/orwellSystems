import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.numeric import False_
import argparse
from pathlib import Path
import ascii_magic
import sys

output = ascii_magic.from_image_file(
    './logo/orwell.png',
    columns=100
)
ascii_magic.to_terminal(output)
print("\nWELCOME TO ORWELL SURVEILLANCE SYSTEMS v0.1\n")
FLAGS = []
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video-path',
                        type=str,
                        help='The path to the video file')
    parser.add_argument('-vo', '--video-output-path',
                        type=str,
                        default='./output.avi',
                        help='The path of the output video file')
    parser.add_argument('-w', '--weights',
                        type=str,
                        default='./yolov4/yolov4_training_last_good.weights',
                        help='Path to the file which contains the weights \
                for YOLOv4.')
    parser.add_argument('-cfg', '--config',
                        type=str,
                        default='./yolov4/yolov4_testing_custom.cfg',
                        help='Path to the configuration file for the YOLOv4 model.')
    parser.add_argument('-l', '--labels',
                        type=str,
                        default='./yolov4/classes_custom.txt',
                        help='Path to the file having the \
                    labels in a new-line seperated way.')
    parser.add_argument('-c', '--confidence',
                        type=float,
                        default=0.2,
                        help='The model will reject boundaries which has a \
                probabiity less than the confidence value. \
                default: 0.2')
    parser.add_argument('-of', '--optical-flow',
                        type=bool,
                        default=False,
                        help='This mode tracks the trajectory \
                of a detected object to and visualizes the track. \
                default: False')

    parser.add_argument('-nn', '--neural-network-size',
                        type=int,
                        default=416,
                        help='Customize the size \
                of the neural network. Multiples of 8. 158 \
                may improve the performance \
                default: 416')
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    FLAGS, unparsed = parser.parse_known_args()
    
    net = cv2.dnn.readNet(FLAGS.weights, FLAGS.config)

    classes = []
    with open(FLAGS.labels, "r") as f:
        classes = f.read().splitlines()
    cap = cv2.VideoCapture(FLAGS.video_path)


    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(200, 3))
    width = int(cap.get(3))
    height = int(cap.get(4))
    _, frame = cap.read()
    if frame is None:
        print("File does not exist!")
        exit(1)
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    out = cv2.VideoWriter(FLAGS.video_output_path,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width*2,height))

    # Lucas kanade params
    lk_params = dict(winSize = (15, 15),
                    maxLevel = 4,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    mask = np.zeros_like(frame)
    # Select the center of a bounding box as the optical flow point
            

    def add_transparent_logo(img1, logo, logo_resize, offset_x, offset_y, thresh, maxVal):
        overlay_img1 = np.ones(img1.shape,np.uint8)*255
        raw = cv2.imread(logo)
        rows,cols,_ = raw.shape
        rows = int(rows*logo_resize)
        cols = int(cols*logo_resize)
        img2 = cv2.resize(raw, (cols, rows), interpolation = cv2.INTER_AREA)
        
        hh,ww,cc = img1.shape
        
        xx = (ww - cols) // 2
        yy = (hh - rows) // 2
        xx = int(xx + xx*offset_x)
        yy = int(yy + yy*offset_y)
        overlay_img1[yy:yy+rows, xx:cols+xx ] = img2

        img2gray = cv2.cvtColor(overlay_img1,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray,thresh,maxVal,cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        temp1 = cv2.bitwise_and(img1,img1,mask = mask_inv)
        temp2 = cv2.bitwise_and(overlay_img1,overlay_img1, mask = mask)
        
        return cv2.add(temp1,temp2)

    def concat_tile(target_image, img_list, empty = False):
        # create new image of desired size and color (blue) for padding
        hh,ww,cc = target_image.shape
        color = (255,0,0)
        result = np.full((hh,ww,cc), color, dtype=np.uint8)
        result = add_transparent_logo(result,'./logo/orwell.png',1,0,0,50,150)

        if empty != True:
            prev_y = 0
            prev_x = 0
            for img in img_list:
                ht,wd,_= img.shape
                # compute center offset
                # xx = (ww - wd) // 2
                # yy = (hh - ht) // 2
                if prev_x > ww:
                    prev_x = 0
                    prev_y = prev_y + ht
                # copy img image into center of result image
                if prev_y+ht < hh and prev_x+wd < ww:
                    result[prev_y:prev_y+ht, prev_x:prev_x+wd] = img
                prev_x = prev_x+wd

        return result

    def crop(img,x,y,bottom,right,label):
        detect  = ['Human head', 'Vehicle registration plate','Person','Car','Weapon']
        if label in detect:
            cropped = img[y:bottom, x:right].copy()
        else:
            cropped = None
        return cropped

    points = []
    ids = []
    id_num = 0
    while True:
        _, img_read = cap.read()
        
        if img_read is None:
            break
        img = img_read
        height, width, _ = img.shape
        
        
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        x = FLAGS.neural_network_size
        blob = cv2.dnn.blobFromImage(img, 1/255, (x, x), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes1 = []
        confidences1 = []
        class_ids1 = []
        cropped_ims = []
        cropped = None
        frame1 = None
        
        for output in layerOutputs:
            new_pt = True
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                    
                if confidence > FLAGS.confidence:
                    
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    bottom = y + h
                    right = x + w

                    # If there is a point in the box, we saw this before
                    for p in points:
                        if p[0] > x and p[0] < x + w and p[1] > y and p[1] < y + h:
                            new_pt = False
                            # update the old one
                            min = 9999
                            target_pt = (0,0)
                            for id in ids:
                                point1 = np.array((x, y))
                                point2 = np.array((id[0][0], id[0][1]))
                                # calculating Euclidean distance
                                # using linalg.norm()
                                dist = np.linalg.norm(point1 - point2)
                                if dist < min:
                                    min = dist
                                    target_pt = [id[0][0],id[0][1]]
                            for id in ids:
                                if id[0] == target_pt:
                                    id[0] = [x,y]
                                    

                    if new_pt == True:
                        points.append([center_x,center_y])
                        ids.append([[x, y],id_num])
                        # put all the points in this 2D array
                        cropped = crop(img,x,y,bottom,right,str(classes[class_id]))
                        if cropped is not None and cropped.size != 0:
                            Path('./output_database/{0}'.format(str(classes[class_id]))).mkdir(parents=True, exist_ok=True)
                            title = './output_database/{0}/id{1}.png'.format(str(classes[class_id]), id_num)
                            cv2.imwrite(title, cropped)
                        id_num = id_num + 1
                    
                    boxes1.append([x, y, w, h])
                    confidences1.append((float(confidence)))
                    class_ids1.append(class_id)
            
            

        indexes = cv2.dnn.NMSBoxes(boxes1, confidences1, 0.2, 0.4)
        if len(indexes)>0:
            for i in indexes.flatten():
                x, y, w, h = boxes1[i]
                label = str(classes[class_ids1[i]])
                cropped = crop(img,x,y,bottom,right,str(classes[class_id]))
                if cropped is not None and cropped.size != 0:
                    cropped_ims.append(cropped)
                
                for id in ids:
                    if x == id[0][0] and y == id[0][1]:
                        curr_id = id[1]
                        break
                    
                confidence = str(round(confidences1[i],2))
                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
                # only crop license plat and head
        if FLAGS.optical_flow == True and len(points) != 0:
            old_points = np.array(points, dtype=np.float32)
            new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
            #old_points = new_points
            points = new_points.tolist()

            for i,(new,old) in enumerate(zip(new_points, old_points)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.arrowedLine(mask, (int(a),int(b)),(int(c),int(d)),(0,255,0), 2,2)
                img = cv2.circle(img,(int(a),int(b)),2,(0,255,0),-1)
            img = cv2.add(img,mask) 
            old_gray = gray_frame.copy()


        frame1 = concat_tile(img,cropped_ims)

        # if frame1 is None:   
        #     frame1 = concat_tile_resize(img,None, empty=True)
        img = add_transparent_logo(img,'./logo/orwell.png',0.5,0.98,-0.9,50,55)
        
        
        both = np.concatenate((img, frame1), axis=1)            
                    
        # cropped_ims.append(img)
        # concat = concat_tile_resize(cropped_ims)
        out.write(both)
        cv2.imshow("Frame", both)
        key = cv2.waitKey(1)
        if key==27:
            cap.release()
            cv2.destroyAllWindows()

    cap.release()
    out.release()

    cv2.imshow("Frame", both)
    cv2.waitKey(0)
    if key==27:
        cap.release()
        cv2.destroyAllWindows()