import cv2
import os


# videofile = '/home/peter/dataset/gist/org/mid2019/gta_jh_trial_1/student1 - 20190722151254.avi'
# txtpath = '/home/peter/dataset/gist/org/mid2019/gta_jh_trial_1/student1 - 20190722151254.txt'

# videofile = '/home/peter/dataset/gist/org/mid2019/roaming_kdh_trial_1/student1 - 20190724165510.avi'
# txtpath = '/home/peter/dataset/gist/org/mid2019/roaming_kdh_trial_1/student1 - 20190724165510.txt'


def trim_txt():
    path = '/home/peter/dataset/gist/org/mid2019/gta_jh_trial_2'
    file = 'ohryong1.txt'
    s, f = 600, 2610

    # path = '/home/peter/dataset/gist/org/mid2019/roaming_kdh_trial_1/student1.txt'

    infile = os.path.join(path, file)
    outfile = os.path.join(path, f'trim_{file}')
    outtxt = open(outfile, 'w')

    bbox_list_list = get_bbox_list(infile)
    selected_list_list = bbox_list_list[s:f]

    for idx, bbox_list in enumerate(selected_list_list):
        for bbox in bbox_list:
            _, cls, x, y, w, h, c = bbox
            aa = f'{idx} {cls} {x} {y} {w} {h} {c}\n'
            outtxt.write(aa)


def get_bbox_list(txtpath):
    txtfile = open(txtpath, 'r')
    txt = txtfile.readlines()

    N = int(txt[-1].split(' ')[0])
    car_list_list = [[] for x in range(N + 1)]
    hm_list_list = [[] for x in range(N + 1)]

    for tx in txt:
        txx = tx.split('\n')[0]
        idx, cls, x, y, w, h, c = txx.split(' ')
        # print(N, idx)
        idx, x, y, w, h, c = int(idx), max(int(x),0), int(y), int(w), int(h), float(c)
        if cls == 'car':
            car_list_list[idx].append([x, y, w, h, c])
        else:
            hm_list_list[idx].append([x, y, w, h, c])

    return car_list_list, hm_list_list


def run():
    path = '/home/peter/dataset/gist/org/mid2019/gta_jh_trial_2'
    file = 'trim_ohryong1'
    videofile = os.path.join(path, f'{file}.mp4')
    txtfile = os.path.join(path, f'{file}.txt')

    cap = cv2.VideoCapture()
    cap.open(videofile)
    idx = 0
    car_list_list, hm_list_list = get_bbox_list(txtfile)

    RED = (0,0,255)
    flag = False
    while 1:
        ret, img = cap.read()
        if ret is False:
            break

        imgW, imgH = img.shape[1], img.shape[0]


        if 10 <idx < 900:

            if flag:
                cv2.rectangle(img, (10, 10), (imgW - 10, imgH - 10), RED, 10)

            if idx%5 ==0:
                flag = not flag
        bbox_list = hm_list_list[idx]
        for bbox in bbox_list:
            # print(bbox)
            x, y, w, h, c = bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (214, 34, 143), 1)

        bbox_list = car_list_list[idx]
        for bbox in bbox_list:
            # print(bbox)
            x, y, w, h, c = bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (99, 255, 143), 1)

        cv2.imshow('1', img)
        cv2.waitKey(15)

        idx = idx + 1


# bbox_list_list = get_bbox_list()
# bbox_list = bbox_list_list[0]
# print(bbox_list)

run()
# trim_txt()
