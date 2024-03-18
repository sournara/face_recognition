import cv2
import time
import numpy as np
import dlib
import math

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
font = cv2.FONT_HERSHEY_SIMPLEX


def first_capture():
    # 1단계 : 무표정일 때 캡처
    vid_o = cv2.VideoCapture(0)

    vid_o.set(3, 640)
    vid_o.set(4, 480)
    while True:
        ret, image_o = vid_o.read()
        image = cv2.resize(image_o, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        cv2.flip(image_o, 1, image)

        text = 'Let\'s start the program! (key = 27)'
        image = cv2.putText(image, text, (10, 450), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow('image', image)
        key = cv2.waitKey(1)

        # if esc,
        if key == 27:
            break

    cv2.imwrite("original.png", image)
    vid_o.release()
    return shape_predictor(image)


def capture(ori_l, ori_r, sec, a, n, level):
    # 2단계 : 표정변화 후 캡처(original 왼쪽, 오른쪽, 몇초후캠끌지, 목표지점얼마나이동할지x, y)
    vid_in = cv2.VideoCapture(0)
    vid_in.set(3, 640)
    vid_in.set(4, 480)
    start_time = time.time()
    ref_time = time.time()

    img_counter = 0
    sum_percent = 0


    while True:
        ret, image_o = vid_in.read()
        image = cv2.resize(image_o, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        cv2.flip(image_o, 1, image)

        landmark = shape_predictor(image)
        blue_l, blue_r, o_distance = blue_point(ori_l, ori_r, landmark[33], a)
        image = cv2.circle(image, tuple(landmark[21]), radius=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        image = cv2.circle(image, tuple(landmark[22]), radius=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        image = cv2.circle(image, tuple(blue_l), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        image = cv2.circle(image, tuple(blue_r), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        level_num = "[Level {}]".format(level)
        nth_round = "[Round {}]".format(n)
        whattime = "Running time: {}".format(round(time.time() - ref_time - 1, 2))
        image = cv2.putText(image, level_num, (10, 20), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        image = cv2.putText(image, nth_round, (10, 40), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        image = cv2.putText(image, whattime, (100, 20), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        glabella = "follow blue point"
        image = cv2.putText(image, glabella, (10, 60), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('image', image)

        key = cv2.waitKey(1)

        if key == 32:
            break
        if (time.time() - start_time >= 0.25) and (time.time() - ref_time >= 1):  # <---- Check if 0.25 sec passed
            dis = distance(landmark[21], landmark[22], blue_l, blue_r)
            percentage = ((o_distance-dis) / o_distance) * 100
            sum_percent = sum_percent + percentage

            img_counter += 1
            start_time = time.time()

            if time.time() - ref_time >= sec:
                break

    vid_in.release()
    average = sum_percent / img_counter

    print("[Round ", n, "] average percentage = ", round(average, 2))
    return average



def shape_predictor(file_name):
    img = file_name
    faces = detector(img)
    face = faces[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, face)
    shape_2d = np.array([[p.x, p.y] for p in shape.parts()])

    return shape_2d


def blue_point(loc_l, loc_r, nose, x):
    b_l = nose - loc_l
    b_r = nose - loc_r
    b_l[0] = b_l[0] + x
    b_l[1] = b_l[1]
    b_r[0] = b_r[0] - x
    b_r[1] = b_r[1]

    original_distance = 2 * x

    return b_l, b_r, original_distance


def time_flow(num):
    return num


original_landmark = first_capture()
location_l = (original_landmark[33] - original_landmark[21])
location_r = (original_landmark[33] - original_landmark[22])


def distance(l, r, b_l, b_r):
    d1 = b_l - l
    d2 = b_r - r
    dis1 = math.sqrt(math.pow(d1[0], 2) + math.pow(d1[1], 2))
    dis2 = math.sqrt(math.pow(d2[0], 2) + math.pow(d2[1], 2))
    return (dis1 + dis2)


def avg(a, c):

    avg1 = capture(location_l, location_r, 4, a, 1, c)
    avg2 = capture(location_l, location_r, 4, a, 2, c)
    avg3 = capture(location_l, location_r, 4, a, 3, c)
    avg4 = capture(location_l, location_r, 4, a, 4, c)
    avg5 = capture(location_l, location_r, 4, a, 5, c)
    avg = (avg1 + avg2 + avg3 + avg4 + avg5) / 5
    return avg


def challenge(x, a):  # x,y =처음 목표까지 거리     # a,b = 목표까지 거리 변화값
    level = 1
    key = cv2.waitKey(1)
    while True:
        result = avg(x, level)
        if key == 9:  # Tab 키 : 프로그램 종료
            print("program stopped!")
            exit(challenge)
        if result >= 20:
            print("success!")
            x = x + a
            level = level + 1
        else:
            print("Fail! try again!")
            x = x - a
            level = level - 1


challenge(3, 3)