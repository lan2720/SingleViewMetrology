# coding:utf-8
import cv2
import numpy as np
import copy

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
        cv2.line(image, tuple(en1[:2]), tuple(en2[:2]), (0,255,0), 1)
    return line 


def translate_point(p, trans):
    # 对二维平面的点进行平移操作
    if p.shape[0] == 3:
        p = p[:2]
    return p+np.array(trans)

def translate_line(l, trans):
    # 对二维平面的直线进行平移操作
    new_l = copy.deepcopy(l)
    new_l[2] = l[2]-l[0]*trans[0]-l[1]*trans[1]
    return new_l

def compute_point_given_line(point, line):
    # p is [None, y, 1] given y or [x, None, 1] given x
    if point[0] is None:
        point[0] = int(-1*np.dot(np.array(point)[1:], line[1:])/line[0])
    elif point[1] is None:
        point[1] = int(-1*np.dot(np.array(point)[[0,2]], line[[0,2]])/line[1])
    else:
        raise Exception("The unknown point has error, point=", p)
    return point

def show_vanishing_point_image(raw_image, vanishing_points, lines, border=30):
    xmin = min([it[0] for it in vanishing_points])
    xmax = max([it[0] for it in vanishing_points])
    ymin = min([it[1] for it in vanishing_points])
    ymax = max([it[1] for it in vanishing_points])
    
    tran_vec = [-xmin+border, -ymin+border]
    # 计算灭点在新图像中的位置
    translated_vanishing_points = []
    for p in vanishing_points:
        newvp = translate_point(p, tran_vec)
        translated_vanishing_points.append(newvp)
    # 原始图片在新图像中的原点位置
    raw_image_new_origin = translate_point(np.zeros(2, dtype=np.int), tran_vec)
    
    # 创建新图像
    new_image = np.ones([(ymax-ymin+1+2*border), (xmax-xmin+1+2*border), 3], dtype=np.uint8)*255
    xr0 = raw_image_new_origin[1]
    xr1 = raw_image_new_origin[1]+raw_image.shape[0]
    yr0 = raw_image_new_origin[0]
    yr1 = raw_image_new_origin[0]+raw_image.shape[1]
    # 将原始图片放置在新图像上
    new_image[xr0:xr1, yr0:yr1, :] = raw_image
    # 绘制3个轴的灭点
    axis = ['x','y','z']
    for p, ax in zip(translated_vanishing_points, axis):
        cv2.circle(new_image, tuple(p.tolist()), 3, (0,0,255), -1)
        cv2.putText(new_image,'VP-%s'%(ax.upper()), tuple((p-10).tolist()), 1, 2, (0,0,0))
    # 绘制得到灭点的直线
    for l in lines:
        new_l = translate_line(l, tran_vec)
        en1 = compute_point_given_line([None, 0, 1], new_l)
        en2 = compute_point_given_line([None, new_image.shape[0], 1], new_l)
        cv2.line(new_image, tuple(en1[:2]), tuple(en2[:2]), (0,255,0), 1)
    cv2.imshow("vanish_point", new_image)
    

def main():
    global points
    cv2.namedWindow("image", 0)
    cv2.resizeWindow("image", 640, 480)
    cv2.setMouseCallback('image', select_point)  # 设置回调函数
    image = cv2.imread("box_small.jpg")

    lines = [np.array([54,103,-13663]),
             np.array([-85,-102,28662]),
             np.array([-60,74,-3112]),
             np.array([-34,75,2630]),
             np.array([-31,8,1332]),
             np.array([-32,0,4608])]
    vanishing_points = [np.array([480,-119,1]), np.array([-215,-132,1]), np.array([144,391,1])] # 存储顺序必须为vpx,vpy,vpz(为方便后面的画图)
    while True:
        for p in points:
            cv2.circle(image, p, 2, (255,0,0), -1)
        for i in range(0,len(points)-1,2):
            e1 = points[i]
            e2 = points[i+1]
            line = line_by_2points(e1, e2, image, vis=True)
            if line.tolist() in [it.tolist() for it in lines]:
                continue
            else:
                print("line:", line)
                lines.append(line)
        for i in range(0, len(lines)-1, 2):
            vp = np.cross(lines[i], lines[i+1])
            vp = (vp/vp[2]).astype(np.int)
            if vp.tolist() in [it.tolist() for it in vanishing_points]:
                continue
            else:
                vanishing_points.append(vp)
                print("vp %d:"%(i/2), vp)
        cv2.imshow("image", image)
        if len(vanishing_points) == 3:
            show_vanishing_point_image(image, vanishing_points, lines)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
