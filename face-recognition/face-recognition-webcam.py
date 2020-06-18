#######################################################################################
############Test code using local webcam using boto3 and Sagemaker endpoint##############
#######################################################################################
import cv2
import boto3
import numpy as np
import json
import imutils


runtime = boto3.client(service_name='runtime.sagemaker')


def mask_prediction(frame, ep):
    b = ''
    sq_img = frame
    success, encoded_frame= cv2.imencode('.jpg', sq_img)
    b = encoded_frame.tobytes()
    endpoint_response = runtime.invoke_endpoint(EndpointName=ep,
                                           ContentType='image/jpeg',
                                           Body=b)
    results = endpoint_response['Body'].read()
    detections = json.loads(results)
    return detections, sq_img


##Actual code

## CHANGE YOUR ENDPOINT NAME HERE
ep = '<ENDPOINT NAME>'
    

vid = cv2.VideoCapture(0)
while(True):
    # Capture the video frame
    # by frame
    (W, H) = (None, None)
    (grabbed, frame) = vid.read()
    frame = imutils.resize(frame, width=1024)
    if not grabbed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        q = W
    detections, final_frame = mask_prediction(frame, ep)
    classes = ['OTHER','IVAN']
    index = np.argmax(detections)
    klass = classes[index]
    cv2.putText(final_frame,klass, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    # Display the resulting frame
    cv2.imshow('Frame', final_frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

