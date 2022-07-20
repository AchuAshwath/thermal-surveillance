import cv2
import numpy as np
import face_detection.face_detection as face_detection
import os
import sys

def Image(path):
	face_detector = face_detection.FaceDetector()
	frame = cv2.imread(path)
	annotated_frame = face_detector.draw(frame)
	cv2.imshow('faces',annotated_frame)
	cv2.waitKey(0)

TEMP_TUNER = 1.80
TEMP_TOLERENCE = 70.6
count = 0
flag = None
def process_frame(frame):
    
    frame = ~frame
    heatmap = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
    
    heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    ret, binary_thresh = cv2.threshold(heatmap_gray, 200, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    image_erosion = cv2.erode(binary_thresh, kernel, iterations=1)
    image_opening = cv2.dilate(image_erosion, kernel, iterations=1)
    
    # Get contours from the image obtained by opening operation
    contours, _ = cv2.findContours(image_opening, 1, 2)

    image_with_rectangles = np.copy(heatmap)


    for contour in contours:
        # rectangle over each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        if (w) * (h) < 2400:
            continue

        # Mask is boolean type of matrix.
        mask = np.zeros_like(heatmap_gray)
        cv2.drawContours(mask, contour, -1, 255, -1)

        # Mean of only those pixels which are in blocks and not the whole rectangle selected
        mean = convert_to_temperature(cv2.mean(heatmap_gray, mask=mask)[0])

        # Colors for rectangles and textmin_area
        temperature = round(mean, 2)
        color = (0, 255, 0) if temperature < 70.6 else (
            255, 255, 127)
        

        # Draw rectangles for visualisation
        image_with_rectangles = cv2.rectangle(image_with_rectangles, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_with_rectangles, "{} C".format(temperature), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    return image_with_rectangles


def whole_frame():
    cap = cv2.VideoCapture(1)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            frame = process_frame(frame)

            cv2.imshow('Thermal', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def process_face(frame):
    
    frame = frame
    heatmap = frame
    
    heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    ret, binary_thresh = cv2.threshold(heatmap_gray, 200, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    image_erosion = cv2.erode(binary_thresh, kernel, iterations=1)
    image_opening = cv2.dilate(image_erosion, kernel, iterations=1)
    

    image_with_rectangles = np.copy(heatmap)
    
    return image_with_rectangles



def convert_to_temperature(pixel_avg):
    """
    Converts pixel value (mean) to temperature depending upon the camera hardware
    """
    f = pixel_avg / TEMP_TUNER
    c = (f - 32) * 5/9

    
    return f

def pixel_to_temperature(pixel):
    temp_min = 80
    temp_max = 110
    pixel_max = 255
    pixel_min = 0
    temp_range = temp_max-temp_min
    temp = (((pixel-pixel_min)*temp_range)/(pixel_max-pixel_min))+temp_min +14
    return temp


def only_face():
    global count,flag
    cap = cv2.VideoCapture(1)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_detector = face_detection.FaceDetector()
    #frame_width = int(cap.get(3))
    #frame_height = int(cap.get(4))
    #out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    i= 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        #frame = cv2.flip(frame, 180)
        if ret == True:
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            output = frame
            faces = face_detector.detect_with_no_confidence(frame)
            if faces ==[]:
                face = False
                flag = 0
            else: 
                face = True
            
            for (x1,y1,x2,y2) in faces:
                if len(faces)>1:
                    count+=1
                roi = output[y1:y2, x1:x2]
                try:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(e)
                    continue

                try:
                    if face == True:
                        if flag ==0:
                            count+=len(faces)
                            flag = 1
                                

                except ValueError:
                    continue
                

                # Mask is boolean type of matrix.
                mask = np.zeros_like(roi_gray)

                # Mean of only those pixels which are in blocks and not the whole rectangle selected
                mean = pixel_to_temperature(np.mean(roi_gray))

                # Colors for rectangles and textmin_area
                temperature = round(mean, 2)
                color = (0, 255, 0) if temperature < 100 else (0, 0, 255)
                

                # Draw rectangles for visualisation
                output = cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output, "{} F".format(temperature), (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)  
                if temperature > 100:
                    try:
                        os.mkdir('persons/person'+str(count))
                    except FileExistsError:
                        pass
                    while(face is True):
                        face = frame[y1+2:y2-1,x1+2:x2-1]
                        cv2.imwrite('persons/person'+str(count)+'/person'+str(count)+'_face'+str(i)+'.jpg', face)
                        print("image captured")
                        i+=1
                
            cv2.imshow('Thermal', output)
            # out.write(output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # whole_frame()
    only_face()
