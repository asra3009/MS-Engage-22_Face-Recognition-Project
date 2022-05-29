from encodings import utf_8
from flask import Flask, render_template, Response
import cv2
import face_recognition
import pandas as pd
import numpy as np
from datetime import datetime




app=Flask(__name__)
camera = cv2.VideoCapture(0)


def attendance(name):
    f=open("attendance.csv", 'r+')
    myDataList = f.readlines()
    nameList = []
    for line in myDataList:
        entry = line.split(',')
        nameList.append(entry[0])
    if name not in nameList:
        time_now = datetime.now()
        tStr = time_now.strftime('%H:%M:%S')
        dStr = time_now.strftime('%m/%d/%Y')
        f.writelines(f'\n{name},{tStr},{dStr}')      




asra_image = face_recognition.load_image_file("dataset/Asra.jpeg")
asra_face_encoding = face_recognition.face_encodings(asra_image)[0]

Narendra_Modi_image = face_recognition.load_image_file("dataset/Narendra Modi.jpg")
Narendra_Modi_face_encoding = face_recognition.face_encodings(Narendra_Modi_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    Narendra_Modi_face_encoding,
    asra_face_encoding
]
known_face_names = [
    "Narendra Modi",
    "Asra"
]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

    
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

    

    # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                attendance(name)
                
                    
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

df = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\flask\\attendance.csv")
df.to_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\flask\\attendance.csv", index=None)

@app.route('/')
def page1():
    return render_template('page1.html')
@app.route('/page2')
def page2():
    return render_template('page2.html')
@app.route('/page3')
def page3():
    return render_template('page3.html')
@app.route('/page4')
def page4():
    return render_template('page4.html')

@app.route('/page5')
def page5():
    data = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\flask\\attendance.csv")
    return render_template('page5.html', tables=[data.to_html()], titles=[''])
@app.route('/page6')
def page6():
    return render_template('page6.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)