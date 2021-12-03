import requests
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
import math
import classica.configs as configs
import random

min_len_shadow = 20


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))
    return rotated

def masking(rows, cols, angle, delta=10, weght_zero=10, type=1):
    crow, ccol = rows // 2, cols // 2
    if type==1:
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - delta:crow + delta, ccol - weght_zero:ccol + weght_zero] = 1
        # mask[crow - 1:crow + 1, ccol - 1:ccol + 1] = 0
    elif type == 2:
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - delta:crow + delta, :] = 1
        mask[crow - delta:crow + delta, ccol-weght_zero:ccol+weght_zero] = 0
    else:
        mask = np.ones((rows, cols, 2), np.uint8)
        mask[crow - delta:crow + delta, ccol-weght_zero:ccol+weght_zero] = 0
    rotated = rotate(mask, angle=angle)


    return rotated

def intersection(o1, p1, o2, p2):
    x = [x - y for x, y in zip(o2, o1)]
    d1 = [x - y for x, y in zip(p1, o1)]
    d2 = [x - y for x, y in zip(p2, o2)]

    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 0.00001:
        return None
    else:
        t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
        r = [i1 + i2 * t1 for i1, i2 in zip(o1, d1)]
        return r

def clustering(img, num_clus=2):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    kernel = 5


    return mask


def rotate_c(p, q, n, adjacency):
    x1, y1 = (p[0], p[1])
    x2, y2 = (q[0], q[1])

    newP = (x2 - x1, y2 - y1)
    fiP = math.atan2(newP[1], newP[0])

    if adjacency == 4:
        fiP += n * math.pi / 2

    else:
        fiP += n * math.pi / 4

    x = math.cos(fiP)
    y = math.sin(fiP)

    eps = 0.001
    if -eps < x < eps:
        x = 0
    elif x < 0:
        x = -1
    else:
        x = 1

    if -eps < y < eps:
        y = 0
    elif y < 0:
        y = -1
    else:
        y = 1

    n1 = int(x1 + x)
    n2 = int(y1 + y)

    return (n1, n2)

def next_pixel(p, q=None, n=1, adjacency=4, maximg=(450,450)):
    if q is None:
        # как будто пришли слева на право
        q = (p[0], p[1]-1)


    while True:
        n1, n2 = rotate_c(p, q, n, adjacency)
        if 0 <= n1 < maximg[0] and 0 <= n2 < maximg[1]:
            break
        n += 1


    return p, q, (n1, n2)



class Shadow():
    def __init__(self, *arg):
        path_img, path_img_out, path_img_out2, path_img_out3, path_img_out4 = arg
        self.path_input_img = path_img
        self.path_img_out = path_img
        self.path_img_out2 = path_img_out2
        self.path_img_out3 = path_img_out3
        self.path_img_out4 = path_img_out4

        self.k4 = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        self.k8 = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]]


    def import_map(self, path_ya, id):
        path_ya = 'https://static-maps.yandex.ru/1.x/?' \
                  'll={},{}' \
                  '&size=450,450' \
                  '&z=18' \
                  '&l=sat'.format(path_ya[1], path_ya[0])

        r = requests.get(path_ya)
        if r.status_code == 200:
            raw = r.content
            with open(self.path_input_img.replace('%', str(id)), 'wb') as f:
                f.write(raw)
            print('ok')
        else:
            print(r.status_code)
            # st.write('Не удалось загрузить карту')

    def plot_image(self, in_out):
        if in_out =='input':
            img = cv.imread(self.path_input_img)
        elif in_out == 'output':
            img = cv.imread(self.path_output_img)
        else:
            return None

        gray = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        plt.figure(figsize=(10,10))
        ax1 = plt.subplot(221)
        ax1.imshow(gray)
        ax1.axis('off')
        ax2 = plt.subplot(222)
        ax2.imshow(gray[:,:,0], cmap='Reds')
        ax2.axis('off')
        ax3 = plt.subplot(223)
        ax3.imshow(gray[:,:,1], cmap='Greens')
        ax3.axis('off')
        ax4 = plt.subplot(224)
        ax4.imshow(gray[:,:,2], cmap='Blues')
        ax4.axis('off')
        plt.show()

    def find_shadow(self, angle, delta_filtr, weght_zero):

        img = cv.imread(self.path_input_img)
        img_r = rotate(img, 0)
        # h,v = img_r.shape[:2]
        # if h < v:
        #     delta = (v - h) // 2
        #     img_r = img_r[:,delta:-delta,:]
        # else:
        #     delta = (h - v) // 2
        #     img_r = img_r[delta:-delta,:,:]


        gray = cv.cvtColor(img_r, cv.COLOR_BGR2YUV)[:,:,0]
        # gray = (255 - gray)

        dft = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = np.log10(1 + cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))


        rows, cols = gray.shape[:2]
        mask = masking(rows, cols, angle, delta=delta_filtr, weght_zero=weght_zero, zeros=True)
        fshift = dft_shift * mask
        magnitude_spectrum_mask = np.log10(1 + cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv.idft(f_ishift)
        img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        img_back = (255*(img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))).astype(np.uint8)
        # gray = gray / np.max(gray)

        ret, thresh = cv.threshold(img_back, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        # opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

        contours, hierarchy = cv.findContours(thresh, 2, cv.CHAIN_APPROX_SIMPLE)

        dc = np.zeros(gray.shape, np.uint8)
        distance = []
        for cnt in contours:
            aver_height = []
            s = cv.contourArea(cnt)
            if 50 < s < 2000:
                hull = cv.convexHull(cnt, returnPoints=True)
                cv.drawContours(dc, [hull], 0, 100, -1)

                leftmost = tuple(hull[hull[:, :, 0].argmin()][0])
                rightmost = tuple(hull[hull[:, :, 0].argmax()][0])
                topmost = tuple(hull[hull[:, :, 1].argmin()][0])
                bottommost = tuple(hull[hull[:, :, 1].argmax()][0])
                o2 = [(leftmost[0] + rightmost[0]) / 2, (topmost[1] + bottommost[1]) / 2]
                k = math.tan(math.radians(90 - angle))
                b = o2[1] - k * o2[0]

                # y = k*x + b
                for ic in range(len(hull)):
                    if ic == 0:
                        o1 = hull[-1][0]
                        p1 = hull[ic][0]
                    else:
                        o1 = hull[ic-1][0]
                        p1 = hull[ic][0]
                    p2 = [0, b]

                    l = intersection(o1, p1, o2, p2)
                    if (l is not None) and (min(o1[0], p1[0]) <= l[0] <= max(o1[0], p1[0]) and
                                            (min(o1[1], p1[1]) <= l[1] <= max(o1[1], p1[1]))):
                            aver_height.append(l)

                if len(aver_height) == 2:
                    x1, y1 = aver_height[0]
                    x2, y2 = aver_height[1]
                    distance.append(math.sqrt((x1-x2)**2+(y1-y2)**2))
                    # print()
                # print(len(aver_height), len(hull))



        plt.figure(figsize=(10,10))

        ax1 = plt.subplot(221)
        ax1.imshow(gray, cmap='Greys_r')
        ax1.axis('off')

        ax3 = plt.subplot(222)
        ax3.imshow(img_back, cmap='Greys')
        ax3.axis('off')

        ax2 = plt.subplot(223)
        ax2.imshow(magnitude_spectrum, cmap='Greys')
        ax2.axis('off')

        ax4 = plt.subplot(224)
        ax4.imshow(magnitude_spectrum_mask, cmap='Greys_r')
        ax4.axis('off')

        plt.savefig(self.path_output_img)

        plt.figure(figsize=(10,10))
        ax = plt.subplot(111)
        # ax.imshow(dc, cmap='Greys')
        # gray = np.clip(gray, 0, 1)
        ax.imshow(np.clip(
            img_r +
                  np.stack([dc, np.zeros(gray.shape), np.zeros(gray.shape)], axis=2)
                  , 0, 255).astype(int))
        ax.axis('off')
        plt.savefig(self.path_img_out2)

        print('среднее значение теней', sum(distance)/len(distance))
        plt.figure(figsize=(5,5))
        ax = plt.subplot(111)
        ax.hist(distance, bins=20)
        # ax.axis('off')
        plt.savefig(self.path_img_out3)

    def find_shadow_gray(self, angle, contours=None):

        img = cv.imread(self.path_input_img)
        img_r = rotate(img, 0)
        gray = self.gray
        if contours is None:
            ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            contours, _ = cv.findContours(thresh, 1, cv.CHAIN_APPROX_SIMPLE)
        else:
            contours = [np.expand_dims(np.array(list(cnt))[:, [1, 0]], 1) for cnt in contours]
        # contours, hierarchy = cv.findContours(gray, 1, cv.CHAIN_APPROX_NONE)

        dc = np.zeros(gray.shape, np.uint8)
        distance = []
        print('total contours {}'.format(len(contours)))

        for cnt in contours:
            aver_height = []
            s = cv.contourArea(cnt)
            if True:
                hull = cv.convexHull(cnt, returnPoints=True)
                # hull = cnt

                leftmost = tuple(hull[hull[:, :, 0].argmin()][0])
                rightmost = tuple(hull[hull[:, :, 0].argmax()][0])
                topmost = tuple(hull[hull[:, :, 1].argmin()][0])
                bottommost = tuple(hull[hull[:, :, 1].argmax()][0])
                o2 = [(leftmost[0] + rightmost[0]) / 2, (topmost[1] + bottommost[1]) / 2]
                k = math.tan(math.radians(90 - angle))
                b = o2[1] - k * o2[0]

                # y = k*x + b
                for ic in range(len(hull)):
                    if ic == 0:
                        o1 = hull[-1][0]
                        p1 = hull[ic][0]
                    else:
                        o1 = hull[ic-1][0]
                        p1 = hull[ic][0]
                    p2 = [0, b]

                    l = intersection(o1, p1, o2, p2)
                    if (l is not None) and (min(o1[0], p1[0]) <= l[0] <= max(o1[0], p1[0]) and
                                            (min(o1[1], p1[1]) <= l[1] <= max(o1[1], p1[1]))):
                            aver_height.append(l)

                if len(aver_height) == 2:
                    x1, y1 = aver_height[0]
                    x2, y2 = aver_height[1]
                    ddd = math.sqrt((x1-x2)**2+(y1-y2)**2)

                    distance.append(ddd)

                    cv.drawContours(dc, [cnt], 0, int(ddd), -1)



        # ax2.imshow(np.clip(
        #     np.stack([gray, gray, gray], axis=2) +
        #           np.stack([dc, np.zeros(gray.shape), np.zeros(gray.shape)], axis=2)
        #           , 0, 255).astype(int))

        maxdc = np.max(dc)
        dc_scal = (dc/maxdc*255).astype(np.uint8)

        plt.figure(figsize=(5,10), frameon=False)

        ax1 = plt.subplot(211)
        ax1.imshow(dc_scal, cmap='YlGnBu')

        ax2 = plt.subplot(212)
        ax2.imshow(dc_scal, cmap='YlGnBu')
        ax2.imshow(img_r, alpha=0.8, )
        ax2.axis('off')

        plt.savefig(self.path_img_out2)
        print(len(distance))
        print('среднее значение теней', sum(distance)/len(distance))
        plt.figure(figsize=(5,5))
        ax = plt.subplot(111)
        ax.hist(distance, bins=20)
        # ax.axis('off')
        plt.savefig(self.path_img_out3)

        return sum(distance)/len(distance), distance

    def porog(self, id, cel):
        img = cv.imread(self.path_input_img)
        # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # img = cv.blur(img, (3, 3))
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # mask = cv.inRange(hsv, lower_blue, upper_blue)
        # lfunc = lambda x: x if np.all(x) > 10 else np.array([255,255,255], dtype=np.uint8)
        # img = lfunc(img)
        data = configs.leps[id]
        contour = np.array(list(data['glade']))

        # в области просеки ищем средний цвет.
        # на одной
        vfunc = np.vectorize(lambda x: np.nan if x == 0. else 1.)
        # mediana = [cv.mean(img[:,:,i], mask = mask)[0] for i in [0, 1, 2]]
        mask_total = np.zeros(img.shape[:2])

        # cel = 15
        new_img = np.zeros(img.shape[:2], dtype=np.uint8)
        mask2 = np.zeros(img.shape[:2])
        cv.drawContours(mask2, [contour], 0, 1, -1)
        # new_img = np.ones(img.shape[:2], dtype=np.uint8)*255
        for jj in range(0, img.shape[1]+1, cel):
            for ii in range(0, img.shape[0]+1, cel):
                mask1 = np.zeros(img.shape[:2])
                cv.circle(mask1,(jj,ii), cel, 1, -1)
                mask = mask1 * mask2
                # mask = mask.transpose()
                mask_total = mask_total + mask
                masknan = vfunc(mask)
                if np.sum(mask) == 0:
                    continue

                # print(ii, jj)
                mediana = [np.nanpercentile(masknan*img[:,:,i], 45) for i in [0, 1, 2]]
                # print([x*10/25 for x in mediana])
                for i in range(max(0, ii - cel), min(ii + cel, img.shape[0])):
                    for j in range(max(0, jj - cel), min(jj + cel, img.shape[1])):
                        if mask[i,j] == 1:
                            v = img[i, j]
                            if sum([v[k] < mediana[k] for k in [0,1,2]]) == 3:
                                if new_img[i, j] != 255:
                                    new_img[i, j] = 0
                            else:
                                new_img[i, j] = 255
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        if mask2[i,j] == 0:
                            new_img[i, j] = 255


        # plt.figure(figsize=(5,10))
        # ax1 = plt.subplot(211)
        # ax1.imshow(new_img, cmap='Greys_r')
        # ax1.axis('off')
        #
        # ax2 = plt.subplot(212)
        # ax2.imshow(mask_total, cmap='Greys')
        # ax2.axis('off')
        # plt.savefig(path_img_out4)

        self.gray = new_img
        return new_img

    def plot_glade(self, id):
        img = cv.imread(self.path_input_img)
        data = configs.leps[id]

        contour = np.array(list(data['glade']))
        cv.drawContours(img, [contour], 0, (0, 255, 0), 1)

        plt.figure(figsize=(6,6))
        ax1 = plt.subplot(111)
        ax1.imshow(img)
        ax1.axis('off')
        plt.show()


    def edit_photo(self, gray, adjacency):
        gray[0, 0] = 255
        w, h = gray.shape[:2]
        gray_new = gray.copy()
        gray_new2 = gray.copy()
        black_points = np.where(gray == 0)
        black_points = list(zip(black_points[0], black_points[1]))
        random.seed(42)
        index_start = 0
        all_contours = []
        while len(black_points) > 0:
            for k in range(index_start, w * h):
                i = k // w
                j = k % w
                if gray_new[i, j] == 0:
                    start = (i, j)
                    rebra = []
                    points_in_square = []
                    points_in_square.append(start)
                    p0 = start
                    first = True
                    while True:
                        if first:
                            p0, q0, q1 = next_pixel(p0, adjacency=adjacency)
                            first = False
                        else:
                            p0, q0, q1 = next_pixel(p0, q1, adjacency=adjacency)

                        try:
                            rebra.index((p0, q1))
                            break
                        except:
                            pass

                        while gray_new[q1[0], q1[1]] == 0:
                            points_in_square.append(q1)
                            p0, q0, q1 = next_pixel(q1, p0, adjacency=adjacency)
                        rebra.append((p0, q1))

                    pc = set(points_in_square)
                    zatirat = False
                    if len(pc) < 40:
                        zatirat = True

                    if not zatirat:
                        all_contours.append(pc)

                    po_for_del = []
                    for ij in black_points:
                        min_x = min([x for x, y in pc])
                        max_x = max([x for x, y in pc])
                        min_y = min([y for x, y in pc])
                        max_y = max([y for x, y in pc])
                        if ij in pc:
                            gray_new[ij[0], ij[1]] = 255
                            po_for_del.append(ij)
                            if zatirat:
                                gray_new2[ij[0], ij[1]] = 255

                        else:
                            x_b = ij[0]
                            y_b = ij[1]
                            if min_x <= x_b <= max_x and min_y <= y_b <= max_y:
                                ch = 0
                                for pp in pc:
                                    # if ch == 4:
                                    #     break
                                    if pp[0] == x_b and pp[1] < y_b:
                                        ch += 1
                                    if pp[0] == x_b and pp[1] > y_b:
                                        ch += 1
                                    if pp[1] == y_b and pp[0] < x_b:
                                        ch += 1
                                    if pp[1] == y_b and pp[0] > x_b:
                                        ch += 1
                                if ch >= 4:
                                    gray_new[ij[0], ij[1]] = 255
                                    po_for_del.append(ij)
                                    if zatirat:
                                        gray_new2[ij[0], ij[1]] = 255

                    for ij in po_for_del:
                        black_points.remove(ij)
                    index_start = k
        return gray_new2, all_contours

    def edit_pixels(self, img1=None, img2=None, adjacency=4):

        if img1 is None:
            gray = self.gray
        else:
            gray = img1

        f1, count1 = self.edit_photo(gray, adjacency)

        if img2 is None:
            plt.figure(figsize=(5, 10))
            ax1 = plt.subplot(211)
            ax1.imshow(gray, cmap='Greys_r')
            ax1.axis('off')
            ax3 = plt.subplot(212)
            ax3.imshow(f1, cmap='Greys_r')
            ax3.axis('off')
            plt.show()
            self.gray = f1
            return f1, count1
        else:
            gray = img2

        f2, count2 = self.edit_photo(gray, adjacency)

        plt.figure(figsize=(5, 20))
        ax1 = plt.subplot(411)
        ax1.imshow(f1, cmap='Greys_r')
        ax1.axis('off')
        ax2 = plt.subplot(412)
        ax2.imshow(f2, cmap='Greys_r')
        ax2.axis('off')

        end_img = np.zeros(img1.shape[:2], dtype=np.uint8)
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                if f1[i, j] == 0 and f2[i, j] == 0:
                    end_img[i, j] = 0
                elif f1[i, j] == 255 or f2[i, j] == 255:
                    end_img[i, j] = 255

        ax4 = plt.subplot(413)
        ax4.imshow(end_img, cmap='Greys_r')
        ax4.axis('off')

        f3, count3 = self.edit_photo(end_img, adjacency)

        ax3 = plt.subplot(414)
        ax3.imshow(f3, cmap='Greys_r')
        ax3.axis('off')
        plt.show()
        self.gray = f3
        return f3, count3


if __name__ == '__main__':
    if False:
        # sourc = st.radio('Источник', ('ya','loc'))
        option = st.selectbox('Пример',
        ('1', '2',))

        if option == '1':
            id = 1
            path_img = './strim_input_1.jpg'
        elif option == '2':
            id = 2
            path_img = './strim_input_2.jpg'


        path_img_out = './data/strim_out.jpg'
        path_img_out2 = './data/stack.jpg'
        path_img_out3 = './data/bar.jpg'
        path_img_out4 = './data/mask_green.jpg'
        sh = Shadow(path_img)
        # if sourc == 'loc':
        #     uploaded_file = st.file_uploader("Choose a file")
        #     if uploaded_file is not None:
        #         bytes_data = uploaded_file.getvalue()
        #         # st.write(bytes_data)
        #
        #         with open(path_img,'wb') as f:
        #             f.write(bytes_data)
        # else:
        #     number1 = st.text_input('долгота в градусах',)
        #     number2 = st.text_input('широта в градусах',)
        #     path_ya = 'https://static-maps.yandex.ru/1.x/?' \
        #                'll={},{}' \
        #                '&size=450,450' \
        #                '&z=17' \
        #                '&l=sat'.format(number1, number2)
        #     if st.button('Закачать'):
        #         sh.import_map(path_ya)

        # weght_filtr = st.number_input('Ширина фильтра', 1, 200, 20, 1)
        # weght_zero = st.number_input('Ширина низких частот', 1, 200, 60, 1)
        angle = st.number_input('Угол', -90, 90, 0, 1)
        best_stt = st.number_input('Радиус окна', 1, 100, 10, 1)
        # sh.find_shadow(angle, weght_filtr, weght_zero)
        adjacency = 8

        if st.button('Найти тени'):
            img1 = sh.porog(id, best_stt)
            img, countours = sh.edit_pixels(img1, adjacency=adjacency)
            sh.find_shadow_gray(angle=angle, contours=countours)
            st.image(path_img_out2)
            st.image(path_img_out3)
        # st.image(sh.path_output_img)
    else:
        option = '1'
        if option == '1':
            id = 1
            path_img = './strim_input_2.jpg'
            default_angle = 10
            default_stt = 20
            default_adjacency = 1
            default_n_maen = 2
        elif option == '2':
            id = 0
            path_img = './strim_input_1.jpg'
            default_angle = 10
            default_stt = 40
            default_adjacency = 0
            default_n_maen = 1

        path_img_out = './strim_out.jpg'
        path_img_out2 = './stack.jpg'
        path_img_out3 = './bar.jpg'
        path_img_out4 = './mask_green.jpg'

        arg = [path_img, path_img_out, path_img_out2, path_img_out3, path_img_out4]
        sh = Shadow(*arg)

        adjacency = default_adjacency

        if default_n_maen == 1:
            stt = default_stt
        else:
            stt = default_stt
            stt2 = default_stt//2

        angle = default_angle
        angle2 = 45

        if default_n_maen == 1:
            img1 = sh.porog(id, stt)
            img, countours = sh.edit_pixels(img1, adjacency=adjacency)
            teni, distance = sh.find_shadow_gray(angle=angle, contours=countours)
        else:
            img1 = sh.porog(id, stt)
            img2 = sh.porog(id, stt2)
            img, countours = sh.edit_pixels(img1, img2, adjacency=adjacency)
            teni, distance = sh.find_shadow_gray(angle=angle, contours=countours)
