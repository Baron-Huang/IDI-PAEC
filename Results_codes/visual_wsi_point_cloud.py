#############################  visiualizing whole-slide image via point cloud ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@polyu.edu.hk
#### Department: Centre for Smart Health, PolyU, Hong Kong
#### Attempt: visiualizing whole-slide image via point cloud

import numpy as np
import cv2
from skimage import io, color
import PIL
import matplotlib.pyplot as plt
import os
import natsort
import re


def draw_point_cloud(points = None, show_mode = False, point_size = 5, save_path = None):
    '''
    :param points:
    :param show_mode:
    :param point_size:
    :param save_path:
    :return:
    '''
    plt.rcParams['font.family'] = 'Arial'
    # 示例数据：每个点包含坐标 + 组别

    # 提取所有组名
    groups = sorted(set(p["Instance cluster"] for p in points))

    # 自动给每个组分配一个颜色
    #rgb_list = [[86/255, 152/255, 196/255, 1],
    #            [10/255, 130/255, 124/255, 1],
    #            [171/255, 130/255, 124/255, 1]]
    colors = plt.cm.tab20(range(len(groups)))
    #colors = rgb_list
    color_map = {group: colors[i] for i, group in enumerate(groups)}

    plt.figure(dpi=300)

    # 绘制不同组的散点
    for g in groups:
        gx = [p["x"] for p in points if p["Instance cluster"] == g]
        gy = [p["y"] for p in points if p["Instance cluster"] == g]
        #names = [p["name"] for p in points if p["group"] == g]
        plt.scatter(gx, gy, s=point_size, label=g, color=color_map[g])

        # 标签放在旁边
        #for (px, py, name) in zip(gx, gy, names):
        #    plt.text(px + 0.05, py + 0.05, name, fontsize=10)

    #plt.xlabel("X")
    #plt.ylabel("Y")
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.axis('off')
    plt.margins(x=0.02, y=0.02)
    plt.gca().invert_yaxis()
    #plt.title("Scatter Plot with Group Colors")
    plt.legend(fontsize=26, markerscale=6, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               handlelength=0.8, columnspacing=0.5, borderaxespad=0)

    if show_mode:
    #plt.grid(True)
        plt.show()
    else:
        pass

    plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0.2)


def visual_wsi_point_cloud(group_arr = None, group_wegt_arr = None, read_path = None,
                           save_path = None, inst_name_list = None, show_mode = True):
    '''
    :param group_arr:
    :param group_wegt_arr:
    :param read_path:
    :param save_path:
    :param inst_name_list:
    :return:
    '''
    patch_name_list = natsort.natsorted(os.listdir(read_path), alg=natsort.ns.PATH)
    patch_xy_list = []
    for name_i in patch_name_list:
        ord_i = re.split('_', name_i[:-4])
        patch_xy_list.append(np.uint8(ord_i[2:]))
    print(patch_name_list)
    print(patch_xy_list[0][1])

    new_pois = []
    for point_i in range(len(patch_xy_list)):
        dict_i = {}
        dict_i['name'] = str(patch_xy_list[point_i][0]) + r'_' + str(patch_xy_list[point_i][1])
        dict_i['y'] = patch_xy_list[point_i][0]
        dict_i['x'] = patch_xy_list[point_i][1]
        dict_i['Instance cluster'] = inst_name_list[group_arr[point_i]]
        new_pois.append(dict_i)
        dict_i = {}

    print(new_pois[0])

    draw_point_cloud(points=new_pois, show_mode=show_mode, point_size=10, save_path = save_path)


def visual_wsi_point_cloud_files(group_arr_list=None, group_wegt_arr_list=None, files_read_path=None,
                                  files_save_path = None, inst_name_mlist=None):
    '''
    :param group_arr:
    :param group_wegt_arr:
    :param files_read_path:
    :param files_save_path:
    :param inst_name_list:
    :return:
    '''
    images_name_list = natsort.natsorted(os.listdir(files_read_path), alg=natsort.ns.PATH)
    print(images_name_list)

    for count_i, patch_name_i in enumerate(images_name_list):
        if not os.path.exists(files_save_path):
            os.mkdir(files_save_path)

        read_path = os.path.join(files_read_path, patch_name_i)
        save_path = os.path.join(files_save_path, patch_name_i + '.jpg')
        visual_wsi_point_cloud(group_arr=group_arr_list[count_i], group_wegt_arr=group_wegt_arr_list[count_i],
                               read_path=read_path, save_path=save_path, inst_name_list=inst_name_mlist[count_i])



if __name__ == '__main__':
    #### single mode
    group_arr = [0 for i in range(200)] + [1 for i in range(300)] + [2 for i in range(357)]
    inst_name_list = ['Other', 'Tumor', 'Non_tumor']
    group_wegt_arr = np.random.rand(857)
    read_path = r'F:\Pytorch_Projects\Visual_WSI\Results\DHMC_0003_delte'
    save_path = r'F:\Pytorch_Projects\Visual_WSI\Results\New_point.jpg'

    visual_wsi_point_cloud(group_arr=group_arr, group_wegt_arr=group_wegt_arr, read_path=read_path,
                           save_path=save_path, inst_name_list=inst_name_list, show_mode=True)



    #### files mode
    #group_arr_list = [[0 for i in range(200)] + [1 for i in range(300)] + [2 for i in range(357)],
    #                  [0 for i in range(200)] + [1 for i in range(300)] + [2 for i in range(357)]]
    #inst_name_mlist = [['Other', 'Tumor', 'Non_tumor'],
    #                   ['Other', 'Tumor', 'Non_tumor']]
    #group_wegt_arr_list = [np.random.rand(857), np.random.rand(857)]
    #files_read_path = r'F:\Pytorch_Projects\Visual_WSI\Results\Segs_delte'
    #files_save_path = r'F:\Pytorch_Projects\Visual_WSI\Results\Visual\Point_map'

    #visual_wsi_point_cloud_files(group_arr_list=group_arr_list, group_wegt_arr_list=group_wegt_arr_list,
    #                             files_read_path=files_read_path, files_save_path = files_save_path,
    #                             inst_name_mlist=inst_name_mlist)
