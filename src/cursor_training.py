from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import random
import dlib
import pickle
from PIL import Image

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

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


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)# loop over the face detections
        rects = detector(gray, 1)
        
        if len(rects) == 0:
                return None, None
        rect = rects[0]
        
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        # detect faces in the grayscale image
                
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        return shape, [shape[30], shape[33], shape[2], shape[14], shape[48], shape[45], shape[8]]


'''
Draws mesh over the image
'''
def draw_mesh(image, shape):
        shape_new = []
        for p in shape:
            shape_new += [[p[0]*2, p[1]*2]]
        shape = shape_new
            
        f = open(",.lines.csv")
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
        

def draw_screens(image, cursor_coordinates, prediction_correct):
    cv2.putText(image, str(cursor_coordinates), (800, 650), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 2);
    # 1920, 1080
    x = cursor_coordinates[0]
    y = cursor_coordinates[1]
    if x < 1920:
        x_rel = 100 + int(x*300/1920)
        y_rel = 50 + int(y*200/1080)
    else:
        x_rel = 450 + int((x-1920)*300/2480)
        y_rel = 50 + int(y*200/1080)
    if x <= 1920:
        cv2.rectangle(image, (100, 50), (400, 250), (255,255,255), -1)
        cv2.rectangle(image, (450, 50), (750, 250), (160,160,160), -1)
    else:
        cv2.rectangle(image, (100, 50), (400, 250), (160,160,160), -1)
        cv2.rectangle(image, (450, 50), (750, 250), (255,255,255), -1)
    cv2.circle(image, (x_rel, y_rel), 4, (0,255,0), -1)
    cv2.putText(image, "Prediction:", (50, 630), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 2);
    if prediction_correct:
        cv2.rectangle(image, (200, 600), (300, 650), (0,255,0), -1)
    else:
        cv2.rectangle(image, (200, 600), (300, 650), (0,0,255), -1)
    return image

prev_points = []
prev_points_num = 3
for i in range(prev_points_num):
    prev_points += [None]

def ave_points(points):
    global prev_points
    for i in range(prev_points_num):
        if prev_points[i] == None:
            prev_points[i] = points

            
    for i in range(prev_points_num-1):
        prev_points[i] = prev_points[i+1]
    prev_points[-1] = points

    p_ave = []
    for i in range(0, len(points)):
        p_sum = 0
        for j in range(prev_points_num):
            p_sum += prev_points[j][i]
        p_ave += [p_sum//prev_points_num]
    return p_ave


'''
Read data file, return all landmark points as well as
just the points used in learning (X), what screen the cursor is on (Y)
and corresponding cursor locations
'''
def get_good_data(filename, points_to_learn):
    all_X = []
    all_Y = []
    cursor_locations = []
    all_mesh_points = []
    image_ids = []
    
    f = open(filename)
    prev_cursor_loc = [0,0]
    num_iterations_cursor_not_moved = 0
    for line in f:
        lst = line.split(",")
        if prev_cursor_loc == [int(lst[136]), int(lst[137])]:
            num_iterations_cursor_not_moved += 1
        else:
            prev_cursor_loc = [int(lst[136]), int(lst[137])]
            num_iterations_cursor_not_moved = 0
        if num_iterations_cursor_not_moved > 5:
            continue
        points = []
        mesh_points = []
        for i in range(0, 135, 2):
            points += [[int(lst[i]), int(lst[i+1])]]
        all_mesh_points += [points]
        #points_to_learn = [
        #        30, # nose tip
        #        33, # nose bottom center
        #        2,  # jawline left
        #        14, # jawline right
        #        48, # mouth corner left
        #        54, # mouth corner right
        #        8   # chin
        #    ]
        #points = ave_points(points)
        X = []
        for i in points_to_learn:
            X += [points[i][0]]
            X += [points[i][1]]


        Y = int(lst[138])

        im_id = int(lst[139])
        image_ids += [im_id]

        all_X += [X]
        all_Y += [Y]
        cursor_locations += [[int(lst[136]), int(lst[137])]]
    return all_X, all_Y, cursor_locations, all_mesh_points, image_ids

def add_jitter(X):
    X_new = []

    for i in range(len(X)):
        x = X[i]
        x_new = []
        for p in x:
            r = random.randint(-1,1)
            x_new += [p + r]
        X_new += [x_new]
    return X_new
    

def get_train_data():
    X, Y, cursor_locations, all_mesh_points = get_good_data("data_2_screens_7800.csv", points)
    X_new = []
    for x in X:
        x_new = []
        for n in x:
            x_new += [n*2]
        X_new += x_new
    return X_new, Y

def stabilize_out(out, n):
    # code for pred stabilization
    prev_pred = out[0]
    prev_pred_used = out[0]
    pred_count = 0

    out_new = []

    for i in range(len(out)):
        # pred_stabilization
        pred = 0
        if out[i] != prev_pred:
            pred_count = 0
        else:
            pred_count += 1
        #print("##################")
        #print("n = "+str(out[i]))
        #print("pred_count = "+str(pred_count))
        #print("prev_pred = "+str(prev_pred))
        
        
        if pred_count >= n:
            pred = out[i]
        else:
            pred = prev_pred_used
        #print("pred = "+str(pred))
        out_new += [pred]
        prev_pred = out[i]
        prev_pred_used = pred
    return out_new

def train_and_test(X, Y, cursor_locations, all_mesh_points, points, image_ids, slvr, alph, hls, rs, config):
    #X = X[:5000] + X[5500:]
    #Y = Y[:5000] + Y[5500:]

    #random.Random(1).shuffle(X)
    #random.Random(1).shuffle(Y)
    #random.Random(1).shuffle(cursor_locations)
    #random.Random(1).shuffle(all_mesh_points)

    #c = list(zip(X, Y, cursor_locations, all_mesh_points))
    #random.shuffle(c)
    #X, Y, cursor_locations, all_mesh_points = zip(*c)


    clf = MLPClassifier(solver=slvr, alpha=alph, hidden_layer_sizes=hls, random_state=rs)

    split_point = 10000

    num_correct = 0
    num_total = 0
    out = []

    X_train = X[split_point:]
    Y_train = Y[split_point:]
    
    print("Number of training samples without jitter = "+str(len(X_train)))

    #X_train_jitter = add_jitter(X_train)
    #X_train = X_train + X_train_jitter
    #Y_train = Y_train + Y_train

    print("Number of training samples with jitter = "+str(len(X_train)))

    # pred with point ave
    #clf.fit(X_train, Y_train)
    #for points in (X[split_point:]):
    #    ave_pts = ave_points(points)
    #    pred = clf.predict([ave_pts])
    #    out += list(pred)

    # normal pred
    clf.fit(X_train, Y_train)
    # save model
    #s = pickle.dumps(clf)
    #clf = pickle.loads(s)
    out = clf.predict(X[:split_point])

    if config["stab_out"]:
        out = stabilize_out(out, 10)

    # pred ensemble
    #out = train_and_test_ensemble(X[0:split_point], Y[0:split_point], X[split_point:])
    
    out_truth = Y[:split_point]
    mesh_points_test = all_mesh_points[:split_point]
    cursor_locations_test = cursor_locations[:split_point]
    image_ids_test = image_ids[:split_point]

    incorrect_points_x = []
    incorrect_points_y = []
    correct_points_x = []
    correct_points_y = []

    # dist from edge histogram
    dist_from_edge = []

    if config["make_video"]:
        video_writer = cv2.VideoWriter(config["video_filename"],cv2.VideoWriter_fourcc(*'MP4V'), 15, (1000, 750))

    for i in range(0,len(out)):
                    
        pred_correct = out[i] == out_truth[i]
        
        if pred_correct:
            num_correct += 1
            correct_points_x += [cursor_locations_test[i][0]]
            correct_points_y += [cursor_locations_test[i][1]]
        else:
            dist_from_edge += [cursor_locations_test[i][0] - 1920]
            incorrect_points_x += [cursor_locations_test[i][0]]
            incorrect_points_y += [cursor_locations_test[i][1]]
        num_total += 1

        if config["make_video"]:
            blank_image = np.zeros((750,1000,3), np.uint8)
            cursor_coordinates = cursor_locations_test[i]
            mesh_img = draw_screens(blank_image, cursor_coordinates, pred_correct)
            mesh_img = draw_mesh(mesh_img, mesh_points_test[i][:70])

        

            plot_size = 50
            if i < plot_size:
                start_idx = 0
            else:
                start_idx = i-plot_size
            if i+plot_size <= len(out):
                end_idx = i+plot_size
            else:
                end_idx = len(out)
            plot_cursor_coord_vs_pred(start_idx, end_idx, cursor_locations_test[start_idx:end_idx], out[start_idx:end_idx], "temp.png", i, 8)
            if i%100 == 0:
                print(str(i)+"/"+str(len(out))+" done.")
            plot_im = cv2.imread("temp.png")
            #cv2.imshow("plot", plot_im)
            #cv2.waitKey(1)
            plot_im = resize(plot_im, width=400, height=250)
            mesh_img[350:600, 580:980] = plot_im

            image_id = image_ids_test[i]
            face_im = cv2.imread("images/image_"+str(image_id)+".png")
            face_im_mesh = cv2.imread("images/image_"+str(image_id)+"_mesh.png")

            mesh_img[350:350+face_im.shape[0], 70:70+face_im.shape[1]] = face_im
            mesh_img[350:350+face_im_mesh.shape[0], 100+face_im.shape[1]:100+face_im.shape[1]+face_im_mesh.shape[1]] = face_im_mesh

            #mesh_img[350:100, 580:980] = plot_im

        
            if not pred_correct:
                #pass
                for i in range(15):
                    video_writer.write(mesh_img)
                #cv2.imshow("mesh", mesh_img)
                #cv2.waitKey(1)
            else:
                #pass
                video_writer.write(mesh_img)
                #cv2.imshow("mesh", mesh_img)
                #cv2.waitKey(1)
                #time.sleep(0.01)

    if config["make_video"]:
        video_writer.release()

    print("Num correct = "+str(num_correct))
    print("Num incorrect = "+str(num_total - num_correct))
    print("Num total = "+str(num_total))
    acc = num_correct*100/num_total
    print("Accuracy = "+str(acc))
    print("#########################")

    if config["plot_scatter"]:
        plt.scatter(correct_points_x, correct_points_y, c="green", alpha=0.5)
        plt.scatter(incorrect_points_x, incorrect_points_y, c="red", alpha=1)
        plt.plot([1920, 1920],[0, max(correct_points_y+incorrect_points_y)])
        plt.title("Correct and incorrect predictions and their locations")
        plt.savefig(config["cursor_loc_plot_filename"])
        plt.show()
    
    if config["plot_dist_from_edge"]:
        plt.hist(dist_from_edge, 20)
        plt.title("Distances of incorrect pred from edge")
        plt.savefig("plot_dist_from_edge.png")
        plt.show()

    if config["plot_lag"]:
        plot_lag(0, len(out), cursor_locations_test, out, config["time_series_plot_filename"], None, 500, config)
    

    return acc

def plot_cursor_coord_vs_pred(start_idx, end_idx, cursor_locations_test, out, plot_filename, curr_idx, plt_size):
    # cursor_x plot
    cursor_x = []
    time = []
    pred = []

    # to track cursor movement
    prev_cursor_coordinates = [0,0]
    num_iter_cursor_not_moved = 0
    nums_iter_cursor_not_moved = []

    for i in range(0, len(out)):
        time += [i+start_idx]
        cursor_coordinates = cursor_locations_test[i]
        cursor_x += [cursor_coordinates[0]]
        if out[i] == 1:
            pred += [1920-500]
        else:
            pred += [1920+500]

        if prev_cursor_coordinates == cursor_coordinates:
            num_iter_cursor_not_moved += 1
        else:
            num_iter_cursor_not_moved = 0
        prev_cursor_coordinates = cursor_coordinates
        nums_iter_cursor_not_moved += [num_iter_cursor_not_moved*100]
    
    # plot cursor x-coordinate and prediction vs time
    plt.figure(figsize=(plt_size,5))
    plt.xlim(start_idx, start_idx+len(out))
    plt.ylim(0, 4600)
    plt.plot(time, cursor_x)
    plt.plot(time, pred, 'r-')
    plt.plot(time, nums_iter_cursor_not_moved)
    plt.plot([0,time[-1]], [1920, 1920], 'k--')

    if curr_idx is not None:
        plt.plot([curr_idx, curr_idx],[0, 4600], "k-")

    plt.savefig(plot_filename)
    plt.close('all')
    #plt.show()


def plot_lag(start_idx, end_idx, cursor_locations_test, out, plot_filename, curr_idx, plt_size, config):
    # cursor_x plot
    cursor_x = []
    time = []
    pred = []

    # to track cursor movement
    prev_cursor_coordinates = [0,0]
    num_iter_cursor_not_moved = 0
    nums_iter_cursor_not_moved = []

    for i in range(0, len(out)):
        time += [i+start_idx]
        cursor_coordinates = cursor_locations_test[i]
        cursor_x += [cursor_coordinates[0]]
        if out[i] == 1:
            pred += [1920-500]
        else:
            pred += [1920+500]

        if prev_cursor_coordinates == cursor_coordinates:
            num_iter_cursor_not_moved += 1
        else:
            num_iter_cursor_not_moved = 0
        prev_cursor_coordinates = cursor_coordinates
        nums_iter_cursor_not_moved += [num_iter_cursor_not_moved*100]
    
    # plot cursor x-coordinate and prediction vs time
    plt.figure(figsize=(plt_size,5))
    plt.xlim(start_idx, start_idx+len(out))
    plt.plot(time, cursor_x)
    plt.plot(time, pred, 'r-')
    plt.plot(time, nums_iter_cursor_not_moved)
    plt.plot([0,time[-1]], [1920, 1920], 'k--')
    
    # plot lag
    lag_values = []
    cursor_locations_x = []
    for point in cursor_locations_test:
        cursor_locations_x += [point[0]]
    prev_cursor_coord_x = None
    prev_pred = None
    i = 0
    while i < len(out):
        pred = out[i]
        cursor_coord_x = cursor_locations_x[i]
        #print("#############################################")
        #print("i = "+str(i))
        #print("prev_pred = "+str(prev_pred))
        #print("pred = "+str(pred))
        #print("cursor_coord_x = "+str(cursor_coord_x))
        #print("prev_cursor_coord_x = "+str(prev_cursor_coord_x))
        if prev_pred == None:
            prev_pred = pred
            prev_cursor_coord_x = cursor_coord_x
        else:
            lag_start = None
            lag_end = None
            lag = None
            if prev_pred == 1 and pred == 2:
                print("found rising pred at "+str(i))
                # rising edge of pred
                # go forward till edge found
                # if rising edge of cursor coord, record, else pass
                lag_start = i
                found, lag_end = find_edge(out, cursor_locations_x, i, "rising", "cursor_coord")
                i = lag_end+1
                if not found:
                    continue
                lag = lag_end-lag_start
                lag_values += [lag]
                print("Lag of "+str(lag)+" found at ["+str(lag_start)+", "+str(lag_end)+"]")
            elif prev_pred == 2 and pred == 1:
                print("found falling pred at "+str(i))
                # falling edge of pred
                # go forward till edge found
                # if falling edge of cursor coord, record, else pass
                lag_start = i
                found, lag_end = find_edge(out, cursor_locations_x, i, "falling", "cursor_coord")
                i = lag_end+1
                if not found:
                    continue
                lag = lag_end-lag_start
                lag_values += [lag]
                print("Lag of "+str(lag)+" found at ["+str(lag_start)+", "+str(lag_end)+"]")
            elif prev_cursor_coord_x <= 1920 and cursor_coord_x > 1920:
                print("found rising cursor coord at "+str(i))
                # rising edge of cursor coord
                # go forward till edge found              
                # if rising edge of pred, record, else pass
                lag_start = i
                found, lag_end = find_edge(out, cursor_locations_x, i, "rising", "pred")
                i = lag_end+1
                if not found:
                    continue
                lag = lag_end-lag_start
                lag_values += [-lag]
                print("Lag of "+str(lag)+" found at ["+str(lag_start)+", "+str(lag_end)+"]")
            elif prev_cursor_coord_x > 1920 and cursor_coord_x <= 1920:
                print("found falling cursor coord at "+str(i))
                # falling edge of cursor coord
                # go forward till edge found
                # if falling edge of pred, record, else pass
                lag_start = i
                found, lag_end = find_edge(out, cursor_locations_x, i, "falling", "pred")
                i = lag_end+1
                if not found:
                    continue
                lag = lag_end-lag_start
                lag_values += [-lag]
                print("Lag of "+str(lag)+" found at ["+str(lag_start)+", "+str(lag_end)+"]")
            else:
                i += 1
            if lag is not None:
                plt.text(i, 1920+250, "lag="+str(lag), fontsize=12)
            plt.plot([lag_start, lag_start],[1920, 1920+200], "k-")
            plt.plot([lag_end, lag_end],[1920, 1920+200], "k-")
            plt.plot([lag_start, lag_end],[1920+200, 1920+200], "k-")
            prev_pred = out[i-1]
            prev_cursor_coord_x = cursor_locations_x[i-1]

    if curr_idx is not None:
        plt.plot([curr_idx, curr_idx],[0, max(cursor_locations_x)], "k-")

    plt.savefig(plot_filename)
    plt.close('all')
    #plt.show()

    total_lag = 0
    for lag in lag_values:
        total_lag += abs(lag)
    print("Total lag = "+str(total_lag))

    plt.hist(lag_values)
    plt.title("Lag values")
    plt.savefig(config["lag_plot_filename"])
    plt.show()

    #plt.hist(lag_values)
    #plt.show()

def find_edge(predictions, cursor_coordinates, start_idx, rising_or_falling, pred_or_cursor_coord):
    start_idx+=1
    print("Looking for "+rising_or_falling+" "+pred_or_cursor_coord+" at "+str(start_idx))
    if pred_or_cursor_coord == "pred":
        if rising_or_falling == "rising":
            idx = start_idx
            for i in range(start_idx, len(predictions)):
                idx = i
                prev_cursor_coord = cursor_coordinates[i-1]
                cursor_coord = cursor_coordinates[i]
                prev_pred = predictions[i-1]
                pred = predictions[i]
                if (prev_pred == 2 and pred == 1) or (prev_cursor_coord <= 1920 and cursor_coord > 1920) or (prev_cursor_coord > 1920 and cursor_coord <= 1920):
                    return False, i
                if prev_pred == 1 and pred == 2:
                    return True, i
            return False, idx
        else:
            idx = start_idx
            for i in range(start_idx, len(predictions)):
                idx = i
                prev_cursor_coord = cursor_coordinates[i-1]
                cursor_coord = cursor_coordinates[i]
                prev_pred = predictions[i-1]
                pred = predictions[i]
                if (prev_pred == 1 and pred == 2) or (prev_cursor_coord <= 1920 and cursor_coord > 1920) or (prev_cursor_coord > 1920 and cursor_coord <= 1920):
                    return False, i
                if prev_pred == 2 and pred == 1:
                    return True, i
            return False, idx
    else:
        if rising_or_falling == "rising":
            idx = start_idx
            for i in range(start_idx, len(cursor_coordinates)):
                idx = i
                prev_cursor_coord = cursor_coordinates[i-1]
                cursor_coord = cursor_coordinates[i]
                prev_pred = predictions[i-1]
                pred = predictions[i]
                if (prev_pred == 1 and pred == 2) or (prev_pred == 2 and pred == 1) or (prev_cursor_coord > 1920 and cursor_coord <= 1920):
                    return False, i
                if prev_cursor_coord <= 1920 and cursor_coord > 1920:
                    return True, i
            return False, idx
        else:
            idx = start_idx
            for i in range(start_idx, len(cursor_coordinates)):
                idx = i
                prev_cursor_coord = cursor_coordinates[i-1]
                cursor_coord = cursor_coordinates[i]
                prev_pred = predictions[i-1]
                pred = predictions[i]
                if (prev_pred == 1 and pred == 2) or (prev_pred == 2 and pred == 1) or (prev_cursor_coord <= 1920 and cursor_coord > 1920):
                    return False, i
                if prev_cursor_coord > 1920 and cursor_coord <= 1920:
                    return True, i
            return False, idx
        

def try_diff_configs():
    points = [
            [
                30, # nose tip
                2,  # jawline left
                14  # jawline right
            ],
            [
                30, # nose tip
                33, # nose bottom center
                2,  # jawline left
                14  # jawline right
            ],
            [
                48, # mouth corner left
                54, # mouth corner right
                8   # chin
            ],
            [
                30, # nose tip
                33, # nose bottom center
                2,  # jawline left
                14, # jawline right
                48, # mouth corner left
                54, # mouth corner right
                8   # chin
            ]
        ]
    solvers=['sgd', 'adam', 'lbfgs']
    alphas = [1e-5, 1e-4, 1e-3]
    hidden_layer_sizes = [
            (7,7,7,7,7,7,7),
            (7,7,7,7,7,7),
            (7,7,7,7,7),
            (8,8,8,8,8,8,8),
            (8,8,8,8,8,8),
            (8,8,8,8,8),
            (9,9,9,9,9,9,9),
            (9,9,9,9,9,9),
            (9,9,9,9,9),
            (10,10,10,10,10,10),
            (10,10,10,10,10),
            (10,10,10,10)
        ]
    f = open("training_results.csv", "w")
    max_acc = 0
    max_acc_config = []
    for p in points:
        print("#"*50)
        print("POINTS = "+str(p))
        for s in solvers:
            print("#"*40)
            print("SOLVER = "+s)
            for a in alphas:
                print("#"*30)
                print("ALPHA = "+str(a))
                for h in hidden_layer_sizes:
                    print("#"*20)
                    print("HIDDEN LAYER SIZE = "+str(h))
                    for r in range(20):
                        print("#"*10)
                        print("RANdOM STATE = "+str(r))
                        acc = train_and_test(p, s, a, h, r)
                        f = open("training_results.csv","a")
                        f.write(".".join(str(p).split(","))+","+s+","+str(a)+","+".".join(str(h).split(","))+","+str(r)+","+str(acc)+"\n")
                        f.close()
                        if acc > max_acc:
                            max_acc = acc
                            max_acc_config = [s, a, h, r]
                        print("$$$ max acc so far = "+str(max_acc)+" with "+str(max_acc_config))

'''
for i in range(20):
    print("#################: "+str(i))
    rf = RandomForestClassifier(n_estimators = 50, random_state = i)
    split_point = 6000
    rf.fit(X[:split_point],Y[:split_point])
    num_correct = 0
    num_total = 0
    out = rf.predict(X[split_point:])
    out_truth = Y[split_point:]

    cursor_locations_test = cursor_locations[:split_point]

    incorrect_points_x = []
    incorrect_points_y = []
    correct_points_x = []
    correct_points_y = []

    for i in range(len(out)):
        if out[i] == out_truth[i]:
            num_correct += 1
            correct_points_x += [cursor_locations_test[i][0]]
            correct_points_y += [cursor_locations_test[i][1]]
        else:
            incorrect_points_x += [cursor_locations_test[i][0]]
            incorrect_points_y += [cursor_locations_test[i][1]]
        num_total += 1

    print("Num correct = "+str(num_correct))
    print("Num total = "+str(num_total))
    acc = num_correct*100/num_total
    print("Accuracy = "+str(acc))
    #train_and_test("adam", 1e-4, (8,8,8,8,8), i)
'''


def train_and_test_with_webcam(X, Y, cursor_locations, all_mesh_points, points, slvr, alph, hls, rs):
    
    
    clf = MLPClassifier(solver=slvr, alpha=alph, hidden_layer_sizes=hls, random_state=rs)

    split_point = 60000

    clf.fit(X[0:split_point], Y[0:split_point])

    # save model
    s = pickle.dumps(clf)
    clf = pickle.loads(s)
    

    cap = cv2.VideoCapture(0)
    while(True):
        print("#################")
        time1 = time.time()
        # Capture frame-by-frame
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        image = resize(image, width=1000)
        time2 = time.time()
        shape, p = get_points(image)
        time3 = time.time()


        if shape is None or p is None:
            cv2.imshow("out", image)
            cv2.waitKey(1)
            print("None")
            continue
        X = []
        for point in p:
            X += [point[0], point[1]]
        print(clf.predict([X]))
        time4 = time.time()
        image = draw_mesh(image, shape)
        time5 = time.time()
        print("Time to get image: "+str(time2-time1))
        print("Time to get landmark points "+str(time3-time2))
        print("Time to get predict: "+str(time4-time3))
        print("Time to get draw mesh: "+str(time5-time4))
        #cv2.imshow("out", image)
        #cv2.waitKey(1)


def main():
    print("\n\n########################################################")
    print("################   START NEW RUN   #####################")
    print("########################################################")
    points = [
                    30, # nose tip
                    33, # nose bottom center
                    2,  # jawline left
                    14, # jawline right
                    48, # mouth corner left
                    54, # mouth corner right
                    8   # chin
                ]

    X, Y, cursor_locations, all_mesh_points, image_ids = get_good_data("data_fast_images_face_50000.csv", points)
    #X2, Y2, cursor_locations2, all_mesh_points2 = get_good_data("data_2_screens_10000.csv", points)
    #X = X1+X2
    #Y = Y1+Y2
    #cursor_locations = cursor_locations1+cursor_locations2
    #all_mesh_points = all_mesh_points1+all_mesh_points2
    print("Number of samples = "+str(len(X)))
    #for i in range(20):
    #    print("########")
    #    print(i)
    #    train_and_test(X, Y, cursor_locations, all_mesh_points, points, "sgd", 0.1e-4, (15,15,15,15,15,15,15,15), i)
    out_dir = "out_50000_1"
    config = {
            "do_avg"                        : False,
            "stab_out"                      : False,
            
            "make_video"                    : True,
            "video_filename"                : out_dir+"/data_visual_test_new_with_face.avi",
            
            "plot_lag"                      : True,
            "lag_plot_filename"             : out_dir+"/plot_lag_histogram.png",
            "time_series_plot_filename"     : out_dir+"/plot_lag_fast.png",
            
            "plot_scatter"                  : True,
            "cursor_loc_plot_filename"      : out_dir+"/plot_correct_incorrect_pred_locations.png",
            
            "plot_dist_from_edge"           : True,
            "dist_from_edge_plot_filename"  : out_dir+"/plot_dist_from_edge.png"
            
        }
    train_and_test(X, Y, cursor_locations, all_mesh_points, points, image_ids, "sgd", 0.2e-4, (15,15,15,15,15,15,15,15), 15, config)

def main2():
    print("\n\n########################################################")
    print("################   START NEW RUN   #####################")
    print("########################################################")
    points = [
                    30, # nose tip
                    33, # nose bottom center
                    2,  # jawline left
                    14, # jawline right
                    48, # mouth corner left
                    54, # mouth corner right
                    8   # chin
                ]

    X, Y, cursor_locations, all_mesh_points, image_ids = get_good_data("data_fast_images_face_50000.csv", points)
    #X2, Y2, cursor_locations2, all_mesh_points2 = get_good_data("data_2_screens_10000.csv", points)
    #X = X1+X2
    #Y = Y1+Y2
    #cursor_locations = cursor_locations1+cursor_locations2
    #all_mesh_points = all_mesh_points1+all_mesh_points2
    print("Number of samples = "+str(len(X)))
    #for i in range(20):
    #    print("########")
    #    print(i)
    #    train_and_test(X, Y, cursor_locations, all_mesh_points, points, "sgd", 0.1e-4, (15,15,15,15,15,15,15,15), i)
    out_dir = "out_50000_stab_1"
    config = {
            "do_avg"                        : False,
            "stab_out"                      : True,
            
            "make_video"                    : True,
            "video_filename"                : out_dir+"/data_visual_test_new_with_face.avi",
            
            "plot_lag"                      : True,
            "lag_plot_filename"             : out_dir+"/plot_lag_histogram.png",
            "time_series_plot_filename"     : out_dir+"/plot_lag_fast.png",
            
            "plot_scatter"                  : True,
            "cursor_loc_plot_filename"      : out_dir+"/plot_correct_incorrect_pred_locations.png",
            
            "plot_dist_from_edge"           : True,
            "dist_from_edge_plot_filename"  : out_dir+"/plot_dist_from_edge.png"
            
        }
    train_and_test(X, Y, cursor_locations, all_mesh_points, points, image_ids, "sgd", 0.2e-4, (15,15,15,15,15,15,15,15), 15, config)
    
def test_stabolize_out():
    x = [0,0,0,1,0,0,0,1,1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,0,0,0,0,0,1,1,0,0]
    x_s = stabilize_out(x, 3)
    print(x)
    print(x_s)

if __name__ == "__main__": 
    main()
    main2()
else:
    print("successfully imported cursor_training")

'''
pseudocode for multi cursor
for every screen:
    last_cursor_coords = None
while True:
    get_screen_looked_at()
    if prev_screen != curr_screen:
         cursor_coord = last cursor coord for the new screen

'''






