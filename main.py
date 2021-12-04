import math
import os
import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import geopandas
from shapely.geometry import Point, Polygon
import random
import json
import matplotlib.pyplot as plt
import cv2 as cv
import torch

region = ['Юг', 'Центр', 'Северо-Запад', 'Средняя Волга', 'Урал', 'Сибирь', 'Восток']

@st.cache
def get_df():
    random.seed(42)
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    rus = geopandas.GeoSeries(world[world['iso_a3'] == 'RUS']['geometry'])
    aa = np.random.randint(-100, 100, size=(2000, 2)) + [95.18068700622543, 63.92040775075484]

    c1 = []
    for i in range(2000):
        s = Point(aa[i, 0], aa[i, 1])
        if rus.contains(s)[18]:
            c1.append([aa[i, 0], aa[i, 1], 100*random.random(), random.choice(region),
                       'ВЛ-'+str(random.randint(1, 1000)), random.randint(2010, 2022), 1])

    df = pd.DataFrame(c1, columns=['lon', 'lat', 'deviation', 'region', 'line', 'year', 'count'])

    return df

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))



level = st.sidebar.radio('Выберите, что хотите посмотреть:', ('Дашборд', 'Алгоритмы'), 0)
if level == 'Дашборд':

    v = st.radio('Ситуация:', ('В целом по РФ', 'Местная'))
    df_big = get_df()
    df = df_big[df_big['year'] == 2021]
    if v == 'В целом по РФ':

        st.title('Состояние просек охранных зон ВЛ ЛЭП РФ')
        st.write('**********************************************************')
        # st.header('Основные показатели KPI:')
        st.title('10% (10190)')
        st.subheader('просек не соответствует нормативу')
        col1, col2 = st.columns(2)
        with col1:
            # st.write('Среднее по региону:')
            d1 = pd.DataFrame(df.groupby(['region'])['deviation'].mean())
            # st.write(d1)
        with col2:
            # st.write('Количество проблемных точек:')
            d2 = pd.DataFrame(df.groupby(['line'])['count'].sum())
            # st.write(d1)
            # st.write(d2[ d2['count'] > 1])
        st.write('**********************************************************')
        st.pydeck_chart(pdk.Deck(
            map_style = 'mapbox://styles/mapbox/light-v9',
                        initial_view_state = pdk.ViewState(
                                            latitude = 63.92040775075484,
                                            longitude = 95.18068700622543,
                                            zoom = 2,
                                            pitch = 0,
                        ),
                        layers =
                            [
                                pdk.Layer(
                                    "HeatmapLayer",
                                    data=df,
                                    opacity=0.9,
                                    get_position=["lon", "lat"],
                                    # threshold=0.75,
                                    aggregation=pdk.types.String("MEAN"),
                                    get_weight="deviation",
                                    auto_highlight=True,
                                    # pickable=True,
                                )
                            ],
         ))
        # st.write(d1)
        # st.write(' ')
        # st.write(' ')
        # st.write('Состояние просек ВЛЭП в ()')
        d1 = d1.sort_values(by=['deviation'], ascending=False)
        fig, ax = plt.subplots()
        ax.bar(height=d1['deviation'], x = d1.index)
        ax.set_title('Состояние просек ВЛ ЛЭП, %')
        plt.xticks(rotation=90)
        st.pyplot(fig)
        # st.line_chart(d1)

    else:
        reg = st.selectbox('Выберите регион', region)
        df1 = df[df['region'] == reg]
        st.write(df1)

        lines_reg = sorted(df1['line'].tolist())
        li = st.selectbox('Выберите ВЛ ЛЭП', lines_reg)

        row = df1[df1['line'] == li].iloc[0]
        n_chan = random.randint(4,10)
        p=[]
        p.append([row["lon"], row["lat"]])

        data_tree = []
        for ii in range(1, n_chan):
            x1, y1 = p[ii-1]
            x2, y2 = [(a * (1 + 0.001*random.random())) if random.random() > 0.5 else (a * (1 - 0.001*random.random()))
                                    for a in (x1, y1)]
            p.append([x2, y2])

            for jj in range(random.randint(200, 400)):
                alfa = random.random()
                xp = alfa * x1 + (1 - alfa) * x2
                yp = y1 + (y2 - y1) / (x2 - x1) * (xp - x1)
                si = random.random()
                if si > 0:
                    yp += 0.0001
                else:
                    yp -= 0.0001

                height1 = random.randint(0, 10)
                if height1 < 3:
                    type = 'low'
                elif height1 < 7:
                    type = 'mean'
                else:
                    type = 'big'


                data_tree.append([ [xp, yp], height1, type])

        df_height = pd.DataFrame(data_tree, columns=['coordinates', 'height', "type"])
        df_height1 = df_height[df_height['type'] == 'low']
        df_height2 = df_height[df_height['type'] == 'mean']
        df_height3 = df_height[df_height['type'] == 'big']
        dic1 = [{"name": li,
                 "color": "#aa11aa",
                 "path": p
                 },]
        json_path = json.dumps(dic1)
        df_path = pd.read_json(json_path)
        df_path["color"] = df_path["color"].apply(hex_to_rgb)
        # print(df_path)

        view_state = pdk.ViewState(latitude=row["lat"], longitude=row["lon"], zoom=9)

        layer_opor = pdk.Layer(
            type="PathLayer",
            data=df_path,
            pickable=True,
            get_color="color",
            width_scale=2,
            width_min_pixels=2,
            get_path="path",
            get_width=2,
        )
        fill_color_low = [[30, 200, 10], [30, 200, 10]]
        fill_color_mean = [[250, 250, 50], [250, 250, 50]]
        fill_color_big = [[200, 20, 20], [200, 20, 20]]

        what_look = st.multiselect('Высота растительности:', ['Низкая', "Средняя", "Высокая"],
                                   ['Низкая', "Средняя", "Высокая"])

        layer_tree1 = pdk.Layer(
                "HeatmapLayer",
                data=df_height1,
                opacity=0.9,
                get_position="coordinates",
                aggregation=pdk.types.String("MEAN"),
                get_weight="height",
                color_range=fill_color_low,
                pickable=True,
                threshold=1,
            )
        layer_tree2 = pdk.Layer(
                "HeatmapLayer",
                data=df_height2,
                opacity=0.9,
                get_position="coordinates",
                aggregation=pdk.types.String("MEAN"),
                get_weight="height",
                color_range=fill_color_mean,
                pickable=True,
                threshold=1,
            )
        layer_tree3 = pdk.Layer(
                "HeatmapLayer",
                data=df_height3,
                opacity=0.9,
                get_position="coordinates",
                aggregation=pdk.types.String("MEAN"),
                get_weight="height",
                color_range=fill_color_big,
                pickable=True,
                threshold=1,
            )
        layer_tree = []
        if 'Низкая' in what_look:
            layer_tree.append(layer_tree1)
        if 'Средняя' in what_look:
            layer_tree.append(layer_tree2)
        if 'Высокая' in what_look:
            layer_tree.append(layer_tree3)

        st.pydeck_chart(pdk.Deck(
                            map_style='mapbox://styles/mapbox/satellite-v9',
                            initial_view_state=view_state,
                            layers = [layer_opor] + layer_tree
             ))

        st.subheader('Предсказание роста ДКП')
        year_f = st.number_input('Год для предсказания', 2022, 2030, 2025)
        # предскажем по 2 точкам
        # df_height = pd.DataFrame(data_tree, columns=['coordinates', 'height', "type"])
        df_height_5 = df_height.copy()
        df_height_5['height_5'] = df_height['height'].apply(lambda row: row / random.randint(2,4) )
        df_height_5['pred'] = df_height_5.apply(lambda row: (row['height'] - row['height_5'])/5 * (year_f-2021) + row['height_5'], axis=1)
        df_height_5['type_new'] = df_height_5['pred'].apply(lambda x: 'low' if x < 3 else ('mean' if x < 7 else 'big'))
        # st.write(df_height_5)
        col1, col2 = st.columns(2)
        with col2:
            st.write('Количество точек с определенной высотой ДКР в {}г'.format(int(year_f)))
            st.write(df_height_5.groupby('type_new')['type'].count())
        with col1:
            st.write(f'Количество точек с определенной высотой ДКР в 2021г')
            st.write(df_height_5.groupby('type')['type_new'].count())

# if height1 < 3:
#     type = 'low'
# elif height1 < 7:
#     type = 'mean'
# else:
#     type = 'big'

else:
    st.title('Используемые алгоритмы:')

    segm_dkr = 'Сегментация наличия ДКР'
    regr_dkr = 'Регрессия высоты ДКР'

    tyalgo = st.radio('', (segm_dkr, regr_dkr, 'Классическое CV', 'Обучение с подкреплением',
                           'Детектор теней и домов нейросеткой', 'Стереопара',
                           'Сегментация по маске (тип дкр)'))

    st.write('**********************************************************************')

    if tyalgo in [segm_dkr, regr_dkr]:
        from unet_lidar.plot_function import plot_function

        if tyalgo == segm_dkr:
            st.write('Бинарная сегментация снимка на лес / не лес.\n'
                     'Каждому пикселю присваивается вероятность быть "лесом" от 0 до 1.\n'
                     'Нейросеть обучена на открытых географических данных Швейцарии.\n'
                     'В качестве ground truth использовались высокоточные лидарные карты местности.')
        else:
            st.write('Регрессия высоты деревьев по снимку.\n'
                     'Каждому пикселю присваивается высота. Диапазон высот - от 0 до 63.75 м.\n'
                     'Нейросеть обучена на открытых географических данных Швейцарии.\n'
                     'В качестве ground truth использовались высокоточные лидарные карты местности.')

        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        images_dir = f'{this_file_dir}/RES/swiss_lidar_and_surface/for_plotting/image'
        image_names = os.listdir(images_dir)
        image_paths = [
            os.path.join(images_dir, name)
            for name in image_names
        ]

        n_im = st.slider('Номер фотографии', 0, 9)
        _, col2, _ = st.columns([1, 3, 1])
        with col2:
            path_img = image_paths[n_im]
            st.image(path_img)

            select = st.radio('Обработка изображения', ('Целиком', 'Блоками по 496 пикселей'))
            is_slice = select != 'Целиком'

            go_unet = st.button('Выполнить')
            if go_unet:
                fig = plot_function(
                    model_name='binary_mask_30_epochs' if tyalgo == segm_dkr else 'regression',
                    file_idx=n_im,
                    is_slice=is_slice
                )
                st.header('Результат')
                st.pyplot(fig)

    if tyalgo == 'Классическое CV':
        from classica.izmeritel_teney import Shadow

        st.header('Пример')
        st.write('Классическими алгоритмами можно искать тени. Тени темнее фона, и если тень хорошо выделяется'
                 ' можно попробовать вырезать ее контур. В сложных ситуациях можно попробовать несколько усредненй.'
                 ' Но тогда время нахождения тени удваивается. Если делать 2 усредненеия, рассчет (не оптимизированный'
                 ' как у нас) может идти до 5 минут. Предлагаем 2 примера. Параметры уже предустановлены, но '
                 'можно ставить свои. Фото были скачены с Яндекса.')
        option = st.selectbox('',
        ('1', '2',))

        if option == '1':
            id = 1
            path_img = './classica/strim_input_2.jpg'
            default_angle = 10
            default_stt = 20
            default_adjacency = 1
            default_n_maen = 0
        elif option == '2':
            id = 0
            path_img = './classica/strim_input_1.jpg'
            default_angle = 10
            default_stt = 40
            default_adjacency = 0
            default_n_maen = 1

        st.image(path_img)
        st.write('Метод предполагает бинаризацию фотографии по скользящему порогу. Соединение полученных точек '
                 'в контура используя разную связность, '
                 'удалению маленьких контуров и измерение длин оставшихся контуров по лучу освещения солнца.'
                 ' Для удаления маленьких контуров (шума) можно использовать дискретное преобразование Фурье '
                 '(в интерфейс не выведено, используется анализ площадей)')

        path_img_out = './classica/strim_out.jpg'
        path_img_out2 = './classica/stack.jpg'
        path_img_out3 = './classica/bar.jpg'
        path_img_out4 = './classica/mask_green.jpg'

        arg = [path_img, path_img_out, path_img_out2, path_img_out3, path_img_out4]
        sh = Shadow(*arg)

        col1, col2 = st.columns(2)
        with col1:
            adjacency = st.selectbox('Связность', (4, 8), default_adjacency)
        with col2:
            n_maen = st.selectbox('Количество итераций усреднения', (1, 2), default_n_maen)
        if n_maen == 1:
            stt = st.number_input('Радиус окна экспозиции', 1, 100, default_stt, 1)
        else:
            col1, col2 = st.columns(2)
            with col1:
                stt = st.number_input('Радиус окна экспозиции', 1, 100, default_stt, 1)
            with col2:
                stt2 = st.number_input('Радиус окна экспозиции', 1, 100, default_stt//2, 1)

        col1, col2 = st.columns(2)
        with col1:
            angle = st.number_input('Угол на солнце', -90, 90, default_angle, 1)
        with col2:
            angle2 = st.number_input('Угол высоты солнца', 0, 90, 45, 1)
        porog = st.number_input('Порог теней для объектов', 1, 100, 50, 1)

        if st.button('Найти тени'):
            if n_maen == 1:
                img1 = sh.porog(id, stt)
                img, countours = sh.edit_pixels(img1, adjacency=adjacency)
                teni, distance = sh.find_shadow_gray(angle=angle, contours=countours)
            else:
                img1 = sh.porog(id, stt)
                img2 = sh.porog(id, stt2)
                img, countours = sh.edit_pixels(img1, img2, adjacency=adjacency)
                teni, distance = sh.find_shadow_gray(angle=angle, contours=countours)

            st.write('Средняя высота теней {:.2f} пикселей, что при угле высоты солнца {}, дает среднюю высоту {:.2f}. '
                     'Если предположить, что 1 пиксель это 0,3м, то средняя высота {:.2f}м. Средняя'
                     ' высота теней не относится к столбам и деревьям. Если использовать порог {:.2f}, то высота столбов'
                     ' окажется {:.2f}м'.format(
                teni, angle2, teni*math.tan(math.radians(angle2)), 0.3*teni*math.tan(math.radians(angle2)), porog,
                0.3*sum([x for x in distance if x > porog])/len([x for x in distance if x > porog])*
                            math.tan(math.radians(angle2))
            ))
            st.write('Найденный тени')
            st.image(path_img_out2)
            st.write('Гистограмма высот')
            st.image(path_img_out3)

    if tyalgo == 'Обучение с подкреплением':
        st.write('ИИ может не только искать на фотографии объекты по примерам из датасета и передавать на постпроцесинг. '
                 'Он может самостоятельно вырабатывать стратегии взаимодействия с "миром". '
                 'ИИ способен сам выработать алгоритм поиска объектов, без примеров, а только получая от нас'
                 ' вознаграждение за свою правильную работу, если мы ее оценили положительно, или научится '
                 ' избегать наказания, если он что то делает не правильно. Например, ИИ '
                 'может сам, без заложенной программы конкретных инструкций, найти на фото 2 красных квадрата и изменить им цвет. Вашему'
                 ' вниманию предлагается робот, который обучался искать квадраты, но мы не говорили ему как он должен '
                 'это делать. Он научился сам. Не всегда он это делает быстро. Чаще всего требуется 20-100 шагов. '
                 'Для увеличения скорости требуется больше времени обучения (без обучения он не решал задачу и за тысячи шагов). '
                 'В более сложных системах, ИИ сможет исправлять человеческие ошибки или работать совместно с другими '
                 'детекторами, корректируя их работу, если он посчитает нужным, исходя из своей стратегии')

        st.write('Координаты первого квадрата')
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.number_input('x', 0, 4, 1)
        with col2:
            y1 = st.number_input('y', 0, 4, 1)
        st.write('Координаты второго квадрата')
        col1, col2 = st.columns(2)
        with col1:
            x2 = st.number_input('x', 0, 4, 3)
        with col2:
            y2 = st.number_input('y', 0, 4, 3)

        if (x1, y1) == (x2, y2):
            st.write('Координаты деревьев должны отличаться')
        else:

            click = st.button('Запустить робота')
            if click:
                from RL.play import Labirint, PolicyNetwork, reiforce
                random.seed(None)
                h = 5
                w = 5
                n_tree = 2
                vis = 5

                n_action = 9
                n_state = (h*w)**(1 + n_tree)
                n_hidden = 256

                env = Labirint(h, w, n_tree, n_state, (x1, y1), (x2, y2))
                # env.render()

                lr = 0.001
                police_net = PolicyNetwork(n_state, n_action, n_hidden, lr)

                reiforce(env, police_net)

                st.image('mygif.gif')

    if tyalgo == 'Детектор теней и домов нейросеткой':
        st.write('Один из самых точных, но при этом требовательных алгоритмов. Обучая сеть на множестве примеров'
                 ', мы можем получить класс объекта или его высоту. В качестве примера мы покажем сеть классификатора'
                 ' Yolo (https://github.com/ultralytics/yolov5), но это не принципиальная архитектура. Можно '
                 'использовать и двухшаговые детекторы или написанные самостоятельно. '
                 'Мы обучили сеть на 300 снимках с космоса выделять тени и дома. Зная длину тени (по пискелям), '
                 'мы классическими методами можем получить высоту объекта. А зная класс объекта - соответственно '
                 'то, что находится в защитной зоне. По тени лэп мы понимаем, где она располагается. '
                 'В целом, в зависимости от датасета, можно получить различные '
                 'классификаторы и регрессоры, которые будут точнее классических методов в большинстве случаев.'
                 )

        st.write('В демо встроены некоторые фотографии, которые были в тестовом датасете. Их можно прогнать через '
                 'детектор и проверить визуально качество детекции')

        list_test = sorted(os.listdir('./data/1.v1i.yolov5pytorch/test/images'))

        n_im = st.slider('Номер фотографии', 0, len(list_test)-1)
        _, col2, _ = st.columns([1, 3, 1])
        with col2:
            path_img = os.path.join('./data/1.v1i.yolov5pytorch/test/images', list_test[n_im])
            st.image(path_img)

        go_yolo = st.button('Выполнить')
        if go_yolo:
            from detect import run
            args = {
            'source': path_img,
            'weights': './yolov5/vesa_for_demo/exp2/weights/best.pt',
            'imgsz': [480, 480],
            'device': 'cpu',
            'save_txt': True,
            'save_conf': True,
            'save_crop': True}

            save_path, save_dir = run(**args)

            st.header('Результат')
            st.write('Найденные объекты:')
            if os.path.exists(os.path.join(save_dir, 'crops')):
                obj_in_yo = os.listdir(os.path.join(save_dir, 'crops'))
                for classiki in obj_in_yo:
                    obj_in_yo2 = os.listdir(os.path.join(save_dir, 'crops', classiki))
                    for classiki2 in obj_in_yo2:
                        col3, col4 = st.columns(2)
                        with col3:
                            st.image(os.path.join(save_dir, 'crops', classiki, classiki2))
                        with col4:
                            st.write(classiki)
            else:
                st.write('...объекты не найдены')

            st.write('Таблица объектов:')
            if os.path.exists(os.path.join(save_dir, 'labels')):
                obj_in_yo = os.listdir(os.path.join(save_dir, 'labels'))
                for filu in obj_in_yo:
                    st.write(pd.read_csv(os.path.join(save_dir, 'labels', filu), delimiter=' ', header=None,
                                         names=('x', 'y', 'w', 'h', 'p')))
            else:
                st.write('...ничего не найдено')

            st.write('Фото')
            st.image(save_path)

    if tyalgo == 'Стереопара':
        st.write('Бинокулярное зрение - реальность. Если решить задачу - определить связанные точки '
                 'на левом и правом видах сцены, то не составляет труда вычислить глубину, т.е. расстояние до '
                 'видимой точки поверхности. Фотография должна быть хорошего качества. '
                 'Имея два изображения и соответствующий математический аппарат можно вычислить диспаратность. '
                 'А по фокусному расстоянию камер, высоты спутника и базы - расстояние.')

        primer = st.radio('Пример', ('Фото с телефона', 'Фото дрона'))

        st.write('Пусть есть два изображения')
        col1, col2 = st.columns(2)
        if primer == 'Фото с телефона':
            with col1:
                st.image('./stereo img/3/l1.jpg')
            with col2:
                st.image('./stereo img/3/r1.jpg')
        else:
            with col1:
                st.image('./stereo img/1/2021-12-03_00-20_2.png')
            with col2:
                st.image('./stereo img/1/2021-12-03_00-21.png')


        st.write('Совместим две фотографии')

        if primer == 'Фото с телефона':
            imgL = cv.imread('./stereo img/3/l1.jpg')
            imgR = cv.imread('./stereo img/3/r1.jpg')
        else:
            imgL = cv.imread('./stereo img/1/2021-12-03_00-20_2.png')
            imgR = cv.imread('./stereo img/1/2021-12-03_00-21.png')


        imgL = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
        imgR = cv.cvtColor(imgR, cv.COLOR_BGR2RGB)

        if primer == 'Фото с телефона':
            (h, w) = imgR.shape[:2]
            (cX, cY) = (h, w)
            M = cv.getRotationMatrix2D((cX, cY), 1, 1.0)
            rotatedR = cv.warpAffine(imgR, M, (w, h))

            M = cv.getRotationMatrix2D((cX, cY), 1, 1.0)
            rotatedL = cv.warpAffine(imgL, M, (w, h))

            img1 = rotatedL[50:500, 50:800]
            img2 = rotatedR[50:500, 51:801]
        else:
            from scipy import ndimage

            rotatedR = ndimage.rotate(imgR, 90)
            rotatedL = ndimage.rotate(imgL, 90)
            img1 = rotatedL[300:1100, 0:1000]
            img2 = rotatedR[309:1109, 0:1000]
        fig, ax = plt.subplots()

        # plt.figure(figsize=(15, 15))

        ax.imshow(img1)
        ax.imshow(img2, alpha=0.5)
        ax.axis('off')

        st.pyplot(fig)

        st.write('Теперь построим первичную карту глубины')

        frame1_new = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        frame2_new = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        stereo = cv.StereoBM_create(numDisparities=128, blockSize=15)
        disparity = stereo.compute(frame1_new, frame2_new)

        fig, ax = plt.subplots()

        ax.imshow(disparity, cmap='turbo')
        ax.axis('off')

        st.pyplot(fig)

        st.write('Если применить фильтры, то получим более сглаженную картину')

        # SGBM Parameters -----------------
        window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

        left_matcher = cv.StereoBM_create(
            numDisparities=128,  # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=15,
        )

        right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

        # FILTER Parameters
        col1, col2 = st.columns(2)
        with col1:
            lmbda = st.number_input('lmbda', 10, 10000, 1000)
        with col2:
            sigma = st.number_input('sigma', 1, 100, 10)
        visual_multiplier = 1.0

        wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        print('computing disparity...')
        displ = left_matcher.compute(frame1_new, frame2_new)  # .astype(np.float32)/16
        dispr = right_matcher.compute(frame2_new, frame1_new)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, frame1_new, None, dispr)  # important to put "imgL" here!!!

        filteredImg = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)

        fig, ax = plt.subplots()

        ax.imshow(filteredImg, cmap='turbo')
        ax.axis('off')
        st.pyplot(fig)

        st.write('Наложим модель диспарантности и правое фото')

        fig, ax = plt.subplots()
        ax.imshow(img2)
        ax.imshow(filteredImg, alpha=0.5, cmap='turbo')
        ax.axis('off')
        st.pyplot(fig)

        st.write('Красный цвет окрашивает близкие объекты, темно синим - дальние. Если спутник будет делать фотографии'
                 ', на которых будут видны движения крон и опор, то можно получить точность определения выстоты +/-1,5м')

    if tyalgo == 'Сегментация по маске (тип дкр)':
        from unet_type_dkr.pyimagesearch import config

        st.write('Нейросети также хорошо могут классифицировать протяженные объекты. Человек может отличить хвойные от '
                 'лиственных, например, на зимней фотографии. Обучим этому сеть (на 35 фото с Яндекса). '
                 'В демо встроены некоторые фотографии, которые были в '
                 'тестовом датасете. Их можно прогнать через детектор и проверить визуально качество детекции')

        imagePaths = open(config.TEST_PATHS2).read().strip().split("\n")

        # list_test = sorted(os.listdir('./data/1.v1i.yolov5pytorch/test/images'))

        n_im = st.slider('Номер фотографии', 0, len(imagePaths) - 1)
        _, col2, _ = st.columns([1, 3, 1])
        with col2:
            path_img = imagePaths[n_im]
            st.image(path_img)
            go_unet = st.button('Выполнить')
            if go_unet:
                from unet_type_dkr.predict import make_predictions, get_unet

                unet = get_unet()

                fig = make_predictions(unet, path_img)
                st.header('Результат')
                st.pyplot(fig)
