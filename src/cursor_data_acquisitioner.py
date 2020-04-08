import pyautogui as p
import scipy
import time
import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = ""#dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

'''
Draws mesh over the image
'''
def draw_mesh(image, shape):
        f = open("./lines.csv")
        for line in f:
                vertices = line.split(",")
                if vertices[0] == 'ï»¿1':
                        vertices[0] = "1";
                v1 = int(vertices[0].strip())
                v2 = int(vertices[1].strip())
                if v1 > 68 or v2 > 68:
                        continue
                x1 = shape[v1-1][0]
                y1 = shape[v1-1][1]
                x2 = shape[v2-1][0]
                y2 = shape[v2-1][1]
                cv2.line(image, (x1,y1), (x2,y2), (255, 255, 255))
        i = 0
        points = [
                30, # nose tip
                33, # nose bottom center
                2,  # jawline left
                14, # jawline right
                48, # mouth corner left
                54, # mouth corner right
                8   # chin
            ]
        for i in points:
                p = shape[i]
                cv2.circle(image, (int(p[0]), int(p[1])), 2, (0,0,255), -1)
                #cv2.putText(image, str(i), (p[0], p[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 2);
                i += 1
        return image

'''
from imutils lib
'''
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

'''
Draws mesh over the image
'''
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def get_points(image):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    faces = detector(gray)
    #print(faces)
    #faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #face = faces[0]
    #face = dlib.dlib.rectangle(face[0], face[1], face[2], face[3])
    if len(faces) == 0:
        return None, None, None, 0, 0, 0, 0
    face = faces[0]

    face_im = image[face.top()-10:face.bottom()+10, face.left()-10:face.right()+10]
                
    shape = predictor(gray, face)
    time1 = time.time()
    shape = shape_to_np(shape)
    #print(time.time()-time1)
    return shape, [shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]], face_im, face.top(), face.bottom(), face.left(), face.right()

'''
Adds one point on each cheek for cmoother morphing
'''
def modify_shape(shape):
        shape_new = []
        for point in shape:
                shape_new += [[point[0], point[1]]]
        x69 = int((shape[2][0] + shape[30][0])/2)
        y69 = int((shape[2][1] + shape[30][1])/2)
        x70 = int((shape[14][0] + shape[30][0])/2)
        y70 = int((shape[14][1] + shape[30][1])/2)
        shape_new += [[x69,y69], [x70,y70]]
        
        return shape_new

'''
Adds corners of the imahe and centers of the edges
to shape
'''
def add_boundary_points(shape, image):
        shape_new = []
        for point in shape:
                shape_new += [[point[0], point[1]]]
                
        height = len(image)
        width = len(image[0])
        shape_new += [[int(width/2),0]]
        shape_new += [[width-1, 0]]
        shape_new += [[width-1, int(height/2)]]
        shape_new += [[width-1, height-1]]
        shape_new += [[int(width/2), height-1]]
        shape_new += [[0, height-1]]
        shape_new += [[0, int(height/2)]]
        shape_new += [[0,0]]

        return shape_new


'''
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, im = cap.read()
    im = do(im)
    cv2.imshow("out", im)
    cv2.waitKey(1)
'''
'''
s = p.size()
width = s.width
height = s.height

for i in range(width//20, width, width//10):
    for j in range(height//20, height, height//10):
        print("("+str(i)+", "+str(j)+")")
        p.moveTo(i, j)
        time.sleep(0.1)
'''
#s = p.size()
#width = s.width
#height = s.height

'''
monitor_info = detect_monitors.monitor_areas()
num_monitors = len(monitor_info)
monitor_dims = []
for i in range(num_monitors):
    print("DETECTING MONITOR "+str(i))
    print("Move cursor to the bottom of screen "+str(i)+", then press enter")
    input()
    y = p.position().y
    print("Move cursor to the right of screen "+str(i)+", then press enter")
    input()
    x = p.position().x
    monitor_dims += [[x, y]]
'''


prev_point = [0,0]
num_iterations_cursor_not_moved = 0

'''
Check if mouse hasnt moved for the last 20 iterations
'''
def check_if_mouse_used(point):
    global num_iterations_cursor_not_moved
    global prev_point
    if point[0] != prev_point[0] or point[1] != prev_point[1]:
        num_iterations_cursor_not_moved = 0
        prev_point = point
        return True
    num_iterations_cursor_not_moved += 1
    if num_iterations_cursor_not_moved > 100:
        prev_point = point
        return False
    prev_point = point
    return True

def main():
        cap = cv2.VideoCapture(0)
        datafile = open("./data_fast_images_face_50000_2.csv", "w")
        num_data_acquired = 0

        while num_data_acquired < 100:
            if num_data_acquired%100 == 0:
                print(str(num_data_acquired)+" data points acquired.")
            #x = random.randint(10, width-10)
            #y = random.randint(10, height-10)
            #p.moveTo(x, y)
            #time.sleep(0.5)
            time00 = time.time()
            ret, image = cap.read()
            image = cv2.flip(image, 1)
            x, y = (0,0)#p.position()

            if not check_if_mouse_used([x,y]):
                continue
            time0 = time.time()
            #print("############################")
            #print("Time to get image: "+str(time0-time00))
            image = resize(image, width=500)
            time1 = time.time()
            #print("Time to resixe image: "+str(time1-time0))
            shape, points, face_im, top, bottom, left, right = get_points(image)
            time2 = time.time()
            #print("Time to get landmark points: "+str(time2-time1))
            if shape is None:
                #print("!!!! No face detected")
                cv2.imshow("out", image)
                cv2.waitKey(1)
                #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                continue
            cv2.imwrite("./images/image_"+str(num_data_acquired)+".png", face_im)
            #shape = modify_shape(shape)
            #shape = add_boundary_points(shape, image)
            mesh = draw_mesh(image, shape)

            face_im_mesh = mesh[top-10:bottom+10, left-10:right+10]
            cv2.imwrite("./images/image_"+str(num_data_acquired)+"_mesh.png", face_im_mesh)
            
            cv2.imshow("out", mesh)
            cv2.waitKey(1)


            line = ""
            for point in shape:
                line += str(point[0])+","+str(point[1])+","
            line += str(x)+","+str(y)
            if x <= 1920:
                line += ",1"
            else:
                line += ",2"
            image_idx = num_data_acquired
            line += ","+str(image_idx)
            datafile.write(line+"\n")
            
            num_data_acquired += 1

            
            #im, rv, tv = do(resize(im, width=1000))
            #cv2.imshow("out", im)
            #cv2.waitKey(1)

        print("Time: "+str(time.time()-t1))

if __name__ == "__main__": 
    main()
else:
    print("successfully imported cursor_data_acquisitioner")

