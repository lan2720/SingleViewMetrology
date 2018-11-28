# coding:utf-8
import cv2
import numpy as np

points = []

def select_point(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        print("get a point: x,y = %d,%d" % (x,y))
        points.append((x,y))


def line_by_2points(pt1, pt2, image=None, vis=False):
    if vis:
        assert image is not None, "please give image to visualize the line"
    line = np.cross(np.array(list(pt1)+[1]), np.array(list(pt2)+[1]))
    if vis:
        en1 = [None, 0, 1]#[image.shape[1]-1,None, 1]
        en2 = [None, image.shape[0], 1]#[None, image.shape[0]-1, 1]
        en1[0] = int(-1*np.dot(np.array(en1)[1:], line[1:])/line[0])
        en2[0] = int(-1*np.dot(np.array(en2)[1:], line[1:])/line[0])
        cv2.line(image, tuple(en1[:2]), tuple(en2[:2]), (0,255,0), 10)
    return line 

cv2.namedWindow("image", 0)
cv2.resizeWindow("image", 640, 480)
cv2.setMouseCallback('image', select_point)  # 设置回调函数
image = cv2.imread("box.jpg")

lines = []
vanishing_points = []
while True:
    for p in points:
        cv2.circle(image, p, 30, (255,0,0), -1)
    for i in range(0,len(points)-1,2):
        e1 = points[i]
        e2 = points[i+1]
        line = line_by_2points(e1, e2, image, vis=True)
        if line.tolist() in [it.tolist() for it in lines]:
            continue
        else:
            lines.append(line)
    for i in range(0, len(lines)-1, 2):
        vp = np.cross(lines[i], lines[i+1])
        #vanishing_points.append(vp)
        print("vp %d:"%(i/2), vp)
    cv2.imshow("image", image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
