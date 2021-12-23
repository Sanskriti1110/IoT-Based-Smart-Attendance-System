import wiotp.sdk.device
import cv2
import os
import time
import datetime
import os,cv2;
import numpy as np
from PIL import Image;
import xlwrite
import firebase_admin;
from firebase_admin import credentials;
from firebase_admin import storage;
import sys
from datetime import datetime
import ibm_boto3
from ibm_botocore.client import Config, ClientError
from ibmcloudant.cloudant_v1 import CloudantV1
from ibmcloudant import CouchDbSessionAuthenticator
from ibm_cloud_sdk_core.authenticators import BasicAuthenticator
import requests




# Constants for IBM COS values
COS_ENDPOINT = "https://s3.jp-tok.cloud-object-storage.appdomain.cloud" 
COS_API_KEY_ID = "q5Rl575Ej4mDUEubYbtxWTU7EZNUBRiBxW3JGjWCmsVK" 
COS_INSTANCE_CRN = "crn:v1:bluemix:public:cloud-object-storage:global:a/07ebc4cd6a9a46bb9844bf93ecf47301:7bb6f356-f9a0-4525-9ef0-5d98e0a8a16d::" 

# Create resource
cos = ibm_boto3.resource("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_INSTANCE_CRN,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

authenticator = BasicAuthenticator('apikey-v2-1x76tp54wlzxvziy003n0hhrvo0art1l5zj7amwvt6u2', '01ecd22c6e9b02ae338e4e07956a29f2')
service = CloudantV1(authenticator=authenticator)
service.set_service_url('https://apikey-v2-1x76tp54wlzxvziy003n0hhrvo0art1l5zj7amwvt6u2:01ecd22c6e9b02ae338e4e07956a29f2@cc4dc272-85d7-450a-a536-30c09d73a555-bluemix.cloudantnosqldb.appdomain.cloud')


EmplIds = {134:{"name":"Ayush Kumar Singh","status":""},
           44:{"name":"Aditya Om","status":""},
           30:{"name":"Shraman Jain","status":""},
           77:{"name":"Aman Mandal","status":""},
           53:{"name":"Sanskriti Binani","status":""}
           }

myConfig = { 
    "identity": {
        "orgId": "d7luey",
        "typeId": "AttendenceSystem",
        "deviceId":"0134"
    },
    "auth": {
        "token": "123456789"
    }
}

face_id = 0
registration_status = ""
recentactivity = ""
count =1

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)



client = wiotp.sdk.device.DeviceClient(config=myConfig, logHandlers=None)
client.connect()


def multi_part_upload_cloudObject(bucket_name, item_name, file_path):
    try:
        print("Starting file transfer for {0} to bucket: {1}\n".format(item_name, bucket_name))
        # set 5 MB chunks
        part_size = 1024 * 1024 * 5

        # set threadhold to 15 MB
        file_threshold = 1024 * 1024 * 15

        # set the transfer threshold and chunk size
        transfer_config = ibm_boto3.s3.transfer.TransferConfig(
            multipart_threshold=file_threshold,
            multipart_chunksize=part_size
        )

        # the upload_fileobj method will automatically execute a multi-part upload in 5 MB chunks for all files over 15 MB
        with open(file_path, "rb") as file_data:
            cos.Object(bucket_name, item_name).upload_fileobj(
                Fileobj=file_data,
                Config=transfer_config
            )

        print("Transfer for {0} Complete!\n".format(item_name))
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to complete multi-part upload: {0}".format(e))




def upload_cloudand(inframe):

    picname=datetime.now().strftime("%y-%m-%d-%H-%M")
    cv2.imwrite(picname+".jpg",inframe)
    multi_part_upload_cloudObject("attendanceintrusion", picname+'.jpg', picname+'.jpg')
    json_document={"link":COS_ENDPOINT+'/'+"attendanceintrusion"+'/'+picname+'.jpg'}
    response = service.post_document(db="employee_attendance", document=json_document).get_result()
    sendSms(COS_ENDPOINT+'/'+"attendanceintrusion"+'/'+picname+'.jpg')
    print(response)
    


def registration(faceid):
    global face_id
    global registration_status
    global recentactivity 
    face_id = faceid
    # Start capturing video
    vid_cam = cv2.VideoCapture(0)

    # Detect object in video stream using Haarcascade Frontal Face
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Initialize sample face image
    count = 0

    assure_path_exists('D:/sanskriti/Projects/DIP Project/project trials/Dataset')
    # Start looping
    while (True):

        # Capture video frame
        _, image_frame = vid_cam.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

        # Detect frames of different sizes, list of faces rectangles
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # Loops for each faces
        for (x, y, w, h) in faces:
            # Crop the image frame into rectangle
            cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Increment sample face image
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            # Display the video frame, with bounded rectangle on the person's face
            cv2.imshow('frame', image_frame)

        # To stop taking video, press 'q' for at least 100ms
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        # If image taken reach 50, stop taking video
        elif count >= 50:
            registration_status = "Image Captured Successfully"
            recentactivity = 'Image Captured for ID '+str(faceid)
            break
    # Stop video
    vid_cam.release()

    # Close all started windows
    cv2.destroyAllWindows()


def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids


def traindata():
    global registration_status
    global recentactivity 
    #calling function to get all images
    faces,Ids = getImagesAndLabels('dataSet')
    #calling function to train data
    s = recognizer.train(faces, np.array(Ids))
    registration_status = "Registration Successful"
    #saving trained data into trial.txt file
    recognizer.write('D:/sanskriti/Projects/DIP Project/project trials/trial.txt')
    recentactivity = 'Registration Successful'





def recognizeImg():
    global count
    global recentactivity
    start = time.time()
    period = 8
    face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0);
    recognizer = cv2.face.LBPHFaceRecognizer_create();
    recognizer.read('D:/sanskriti/Projects/DIP Project/project trials/trial.txt');
    id = 0;
    filename = 'filename';
    dict = {
        'item1': 1
    }
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, img = cap.read();
        if(img is None):
            break;
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
            faces = face_cas.detectMultiScale(gray, 1.3, 7);
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2);
                id, conf = recognizer.predict(roi_gray)
                check,frame=cap.read()
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                if (conf <50):

                    a = list(EmplIds.keys())
                    if id in a:
                        status = EmplIds[id]["status"]
                        if status != "PRESENT":
                            name = EmplIds[id]["name"]
                            if ((str(id)) not in dict):
                                filename = xlwrite.output('attendance', 'class1', count, name, 'yes',current_time);
                                count=count+1
                                EmplIds[id]["status"]= "PRESENT"
                                recentactivity = str(id)+" Entered"
                                break
                            
                else:
                    video=cv2.VideoCapture(0)
                    check,frame=video.read()
                    frame = cv2.resize(frame, (600,400))
                    upload_cloudand(frame)
                    id = 'Unknown, can not recognize'
                    recentactivity = 'Unrecognized person detected.'
                    break

                cv2.putText(img, str(id) + " " + str(conf), (x, y - 10), font, 0.55, (120, 255, 120), 1)
            cv2.imshow('frame', img);
            if time.time() > start + period:
                break;
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break;

    cap.release();
    cv2.destroyAllWindows();


def sendSms(linktoimg):
        url = 'https://www.fast2sms.com/dev/bulkV2'
        message = 'Intrusion has been detected please click on the link to view. '+' '+linktoimg
        numbers = '9772124202'
        payload = f'sender_id=TXTIND&message={message}&route=v3&language=english&numbers={numbers}'
        headers = {
            'authorization':'XZi0uN3hcxTtUnkD7sl9gEo2VARfrjQwGbOW1IvyaP68pL54dJvsR8TQ3mw7N4aCx2dhVOtqEL5KADBp',
            'Content-Type':'application/x-www-form-urlencoded'
            }
        response = requests.request("POST",url=url,data=payload, headers=headers)
        print(response.text)



def myCommandCallback(cmd):
    m = cmd.data['command']
    if(isinstance(m, int)):
        registration(m)
    elif(m == "Register"):
        traindata()
    elif(m == "recognize"):
        recognizeImg()
        

while True:
    myData={'regsts':registration_status,'recact':recentactivity}
    client.publishEvent(eventId="status", msgFormat="json", data=myData, qos=0, onPublish=None)
    print("Published data Successfully: %s", myData)
    client.commandCallback = myCommandCallback
    time.sleep(2)




