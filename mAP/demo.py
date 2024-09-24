import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
from collections import defaultdict
import numpy as np


import sys 

# sys.stdout = open("../output.txt","w")
################################
"""
定义函数，用于比较两个文件夹中的文件差异
使用 os.listdir 获取文件夹中的文件列表，并转换为集合
# 找出两个文件夹中不同的文件
如果差异文件是文本文件（以 .txt 结尾），则在第一个文件夹中创建一个空的文本文件
创建文件并打印出创建的空文件路径。
"""
def compare_folders(folder1, folder2):
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    difference = files1.symmetric_difference(files2)

    for file_name in difference:
        if file_name.endswith(".txt"):
            # Create an empty .txt file in the first folder
            empty_file_path = os.path.join(folder1, file_name)
            with open(empty_file_path, 'w'):
                pass
            print(f"Empty file created: {empty_file_path}")

# 默认的最小重叠阈值，用于判断检测框是否匹配（）
MINOVERLAP = 0.25 # default value (defined in the PASCAL VOC2012 challenge)


#########################
"""
创建一个解析命令行参数的 ArgumentParser 对象
添加命令行参数，用于控制是否显示动画、绘图或简化控制台输出
添加参数以忽略一个或多个类别
添加参数以设置特定类别的 IoU 阈值
"""
parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
# argparse receiving list of classes to be ignored (e.g., python main.py --ignore person book)
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
# argparse receiving list of classes with specific IoU (e.g., python main.py --set-class-iou person 0.7)
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")

##################################
"""
添加命令行参数，用于指定 ground-truth 文件夹、检测结果文件夹、图像文件夹和输出文件夹的路径
"""
parser.add_argument("--ground-truth", required=True, default="stain_48", help="Root folder path")
parser.add_argument("--detection-results", required=True, default="hunter", help="Root folder path")
parser.add_argument("--images-folder", required=True, default="stain_48", help="Root folder path")
parser.add_argument("--output-path", required=True, default="stain_48", help="output path")
# 解析命令行输入的参数。
args = parser.parse_args()

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''
###########################
"""
如果没有提供要忽略的类别，则设置为空列表
检查是否设置了特定类别的 IoU 阈值
将当前工作目录更改为脚本所在的目录，以确保所有路径都是正确的
"""
# if there are no classes to ignore then replace None by empty list
if args.ignore is None:
    args.ignore = []

specific_iou_flagged = False
if args.set_class_iou is not None:
    specific_iou_flagged = True

# make sure that the cwd() is the location of the python script (so that every path makes sense)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#########################
"""
定义一个变量 root_folder
定义 ground-truth 文件夹的路径，使用 os.path.join 来连接根路径和用户指定的路径
定义检测结果文件夹的路径
定义输出文件夹的路径，用于存放结果和图表
定义存放图像的文件夹路径
"""
root_folder = ''
GT_PATH = os.path.join('/workspaces/app/mAP/ground-truth', args.ground_truth)
DR_PATH = os.path.join('/workspaces/app/mAP/detection-results', args.detection_results)
output_files_path = os.path.join("/workspaces/app/mAP/output", args.output_path)
# if there are no images then no animation can be shown
IMG_PATH = os.path.join('/workspaces/app/mAP/images', args.images_folder)

#############################
"""
使用 glob 模块获取检测结果文件夹中所有 txt 文件的列表
使用 glob 模块获取 ground-truth 文件夹中所有 txt 文件的列表
如果检测结果的文件数量和 ground-truth 的文件数量不一致，调用 compare_folders 函数比较两个文件夹并处理差异
"""
dr_files_list = glob.glob(DR_PATH + '/*.txt')
ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')

if len(dr_files_list) != len(ground_truth_files_list):
            compare_folders(DR_PATH, GT_PATH)

if os.path.exists(IMG_PATH): 
    for dirpath, dirnames, files in os.walk(IMG_PATH):
        if not files:
            # 如果图像文件夹存在但没有文件，则设置参数以跳过动画显示
            args.no_animation = True
else:
    args.no_animation = True
#如果图像文件夹不存在，也设置参数以跳过动画显示

#######################
"""
初始化一个变量，用于控制是否显示动画
尝试导入 OpenCV 库，如果成功，则设置 show_animation 为 True。如果失败，则打印错误信息并设置参数以跳过动画显示
"""
# try to import OpenCV if the user didn't choose the option --no-animation
show_animation = False
if not args.no_animation:
    try:
        import cv2
        show_animation = True
    except ImportError:
        print("\"opencv-python\" not found, please install to visualize the results.")
        args.no_animation = True

#############################
"""
初始化一个变量，用于控制是否绘制图表
尝试导入 Matplotlib 库，如果成功，则设置 draw_plot 为 True。如果失败，则打印错误信息并设置参数以跳过图表绘制
"""

# try to import Matplotlib if the user didn't choose the option --no-plot
draw_plot = False
if not args.no_plot:
    try:
        import matplotlib.pyplot as plt
        draw_plot = True
    except ImportError:
        print("\"matplotlib\" not found, please install it to get the resulting plots.")
        args.no_plot = True

#############################
"""
定义log_average_miss_rate函数，计算 log-average miss rate (LAMR)，这是评估目标检测算法性能的一个指标
它接受三个参数：precision（精度）、recall（召回率）和 num_images（图像数量）。
该函数用于衡量在不同 false positive per image (fppi) 水平下的 miss rate，并计算它们的对数平均值，从而得到一个综合的性能指标
"""
def log_average_miss_rate(prec, rec, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    函数的文档字符串（docstring）提供了关于 log-average miss rate 的计算方法和输出的说明，以及相关文献的引用。
    """

    # 如果某个类别没有检测到任何对象if there were no detections of that class
    """
    如果精度向量 prec 的长度为 0（即没有检测到该类别的任何对象），
    则将 log-average miss rate (lamr) 设置为 0，miss rate (mr) 设置为 1（表示所有 ground truth 对象都被错过了），
    false positives per image (fppi) 设置为 0，并返回这些值。
    """
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi
    
    """
    计算 false positives per image (fppi) 和 miss rate (mr)。fppi 是 1 减去精度，mr 是 1 减去召回率。
    """
    fppi = (1 - prec)
    mr = (1 - rec)
    #在 fppi 和 mr 的数组前分别插入 -1.0 和 1.0，这是为了在接下来的计算中使用
    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    #使用 NumPy 的 logspace 函数在对数空间中创建 9 个均匀分布的参考点，这些点的值从 10^-2 到 10^0
    ref = np.logspace(-2.0, 0.0, num = 9)
    #对于每个参考点，找到 fppi_tmp 中大于或等于该参考点的第一个元素的索引，并使用这个索引从 mr_tmp 中获取对应的 miss rate 值
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    #由于对数函数在 0 处没有定义，使用 np.maximum 函数确保所有参考点的值都不会小于 1e-10。然后计算这些参考点的对数，取平均值，并用 math.exp 计算其指数，得到 log-average miss rate
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi

###############################################
"""
 throw error and exit
 定义了error函数，接受一个参数 `msg`，用于在程序中出现错误时输出错误信息并终止程序执行
 
 sys.exit() 函数用于立即终止程序执行。它接受一个可选的退出状态码，在这里传递了 0 表示程序因为遇到错误而正常退出。在大多数操作系统中，非零的退出状态码通常表示程序异常终止，而状态码 0 表示程序成功执行完毕。
 使用 error 函数可以在程序中遇到不可控的错误时，给用户一个清晰的错误提示，并且立即停止程序的进一步执行，防止错误进一步扩散。
"""
def error(msg):
    print(msg)
    sys.exit(0)
#调用 sys 模块的 exit 函数，并向其传递一个状态码 0

###################################################
"""
 check if the number is a float between 0.0 and 1.0
 定义函数 is_float_between_0_and_1，接受一个参数 `value`,用于检查传入的值是否为一个介于 0.0 和 1.0 之间的浮点数
"""
def is_float_between_0_and_1(value):
    try:
        val = float(value)
        # 尝试将输入的 `value` 转换为浮点数，并将其存储在变量 `val` 中
        if val > 0.0 and val < 1.0:
            return True
        # 如果 `val` 大于 0.0 并且小于 1.0，返回 `True`
        else:
            return False
        # 如果 `val` 不在指定范围内，返回 `False`
    except ValueError:
        return False
    # 如果 `value` 不能转换为浮点数（例如，如果它是一个无法解析为数字的字符串），则捕获 `ValueError` 并返回 `False`


#################################################
"""
定义 voc_ap 函数，接受两个参数：召回率数组 `rec` 和 精度数组 `prec`用于计算目标检测中的平均精度（Average Precision, AP）
 ###Calculate the AP given the recall and precision array
  函数的目的：给定召回率（recall）和精度（precision）数组，计算平均精度（AP）
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
         首先，计算一个精度单调递减的测量精度/召回率曲线；
    2nd) We compute the AP as the area under this curve by numerical integration.
         其次，通过数值积分计算曲线下面积作为AP###
"""
#该函数通过数值积分的方式计算精度/召回率曲线下的面积，得到平均精度
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    #在召回率数组的开始插入0.0，在末尾插入1.0，然后复制这个数组
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    #在精度数组的开始插入0.0，在末尾插入0.0，然后复制这个数组
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
        #循环从 mpre 数组的倒数第二个元素开始到第一个元素，确保每个元素不小于它后面的元素，实现精度的单调递减,使精度数组从末尾到开头单调递减
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
            #创建一个空列表 i_list，然后遍历 mrec，如果当前元素与前一个元素不同，则将当前索引添加到 i_list,其中的元素是召回率发生变化的位置
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
        # 计算 AP，即曲线下面积的数值积分
    return ap, mrec, mpre
    #返回计算得到的 AP 值以及修改后的召回率和精度数组

##################################
"""
 Convert the lines of a file to a list
 定义了一个名为 file_lines_to_list 的函数，接受一个参数 `path`，这个参数预期是一个文件路径，其作用是将文件的每一行读取到一个列表中
"""
def file_lines_to_list(path):
    # open txt file lines to a list
    """
    使用 with 语句安全地打开文件，open(path) 根据提供的 path 打开文件。f.readlines() 读取文件的所有行，并返回一个列表，其中每个元素是文件的一行
    """
    with open(path) as f:
        content = f.readlines()
        # 使用上下文管理器打开文件，并调用 `readlines` 方法读取文件的所有行，将它们作为一个列表赋值给变量 `content`
    # remove whitespace characters like `\n` at the end of each line
    #使用列表推导式和 `strip` 方法移除每行末尾的空白字符，如换行符 `\n`
    #通过列表推导式处理 content 列表，x.strip() 移除每个元素（即每行文本）末尾的空白字符
    content = [x.strip() for x in content]
    return content
    #返回处理后的列表，其中包含文件的每行，没有末尾的空白字符

"""
 Draws text in image
 定义了 draw_text_in_image 函数，它接受五个参数：
    # img: 要在其上绘制文本的图像
    # text: 要绘制的文本字符串
    # pos: 文本绘制的起始位置（左下角）
    # color: 文本颜色的 BGR 值（例如，白色为 (255, 255, 255)）
    # line_width: 绘制文本的线宽,其作用是在图像上绘制文本
"""
def draw_text_in_image(img, text, pos, color, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    # 设置字体为 OpenCV 的 FONT_HERSHEY_PLAIN 样式，字体缩放比例为 1，线条类型为 1
    bottomLeftCornerOfText = pos
    #将 pos 参数赋值给 bottomLeftCornerOfText，表示文本绘制的起始位置为左下角
    cv2.putText(img, text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            color,
            lineType)
    #使用 OpenCV 的 putText 函数在图像 img 上绘制文本 text。函数的其他参数指定了文本的位置、字体、缩放比例、颜色和线条类型
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    #使用 OpenCV 的 getTextSize 函数计算文本的宽度和高度，这里只关心宽度，因此高度用 _ 忽略
    return img, (line_width + text_width)
    #函数返回更新后的图像 img 和文本宽度加上传入的 line_width，用于确定在同一图像上绘制下一行文本的起始位置


################################################
"""
 Plot - adjust axes
 定义了 adjust_axes 函数，其接受四个参数：
    # r: 渲染器对象，用于测量文本
    # t: 要测量的文本对象
    # fig: Matplotlib 图形对象
    # axes: Matplotlib 坐标轴对象，主要用于调整绘图中的坐标轴尺寸，确保文本标签在图中完整显示
"""
def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling，调用 get_window_extent 方法获取文本对象 t 的边界框，这包括文本的尺寸和位置
    bb = t.get_window_extent(renderer=r)
    # get axis width in inches，将边界框的宽度 bb.width 除以图形的 DPI（dots per inch，每英寸点数）fig.dpi，得到文本宽度的英寸表示
    text_width_inches = bb.width / fig.dpi
    #使用 get_figwidth 方法获取当前图形对象 fig 的宽度
    current_fig_width = fig.get_figwidth()
    #将当前图形宽度与文本宽度相加，得到新的图形宽度
    new_fig_width = current_fig_width + text_width_inches
    #计算新图形宽度与当前图形宽度的比例，用于调整坐标轴的尺寸
    propotion = new_fig_width / current_fig_width
    # get axis limit，使用 get_xlim 方法获取当前坐标轴 axes 的 x 轴界限
    x_lim = axes.get_xlim()
    #使用 set_xlim 方法设置新的 x 轴界限。新界限的第一个值保持不变，第二个值将原始界限的第二个值乘以之前计算的比例，从而扩展 x 轴的范围，确保文本不会超出图形边界
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])

########################################################

'''
draw precision and recall 
定义 create_precision_recall_table 函数，接受五个参数：每个类别的 ground truth 计数、检测计数、类别数量、行高、列宽和保存路径
其作用是绘制一个表格来展示每个类别的精度（precision）和召回率（recall）
'''
def create_precision_recall_table(gt_counter_per_class,det_counter_per_class,
        n_classes,row_height=0.2, column_width=0.3, save_path=None):
    print(f"gt_counter_per_class:{gt_counter_per_class}")
    print(f"det_counter_per_class:{det_counter_per_class}")
    print(f"n_classes:{n_classes}")
    
    #按检测计数字典中的值（即检测到的对象数量）进行排序
    sorted_dic_by_value = sorted(det_counter_per_class.items(), key=operator.itemgetter(0))
    # print(f"sorted_dic_by_value:{sorted_dic_by_value}");

    # 将排序后的项分为两个列表：sorted_keys（类别名称）和 sorted_values（对应的检测计数）
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    # print(f"sorted_keys:{sorted_keys}");
    
    """
    始化两个列表 fp_sorted 和 tp_sorted，用于存储每个类别的假正例和真正例的数量。
    这里的 true_p_bar 是一个已经定义的变量，包含每个类别的真正例计数
    """
    fp_sorted = []
    tp_sorted = []
    for key in sorted_keys:
        fp_sorted.append(det_counter_per_class[key] - true_p_bar[key])
        tp_sorted.append(true_p_bar[key])
    # print(f"fp_sorted:{fp_sorted}")
    # print(f"tp_sorted:{tp_sorted}") 
    
    # 创建三个空列表，用于存储类别名称、计算得到的精度和召回率   
    categories = []
    precisions = []
    recalls    = []
    
    #对 ground truth 计数字典按值排序，并将其项分为两个列表：gtsorted_keys（类别名称）和 gtsorted_values（对应的 ground truth 计数）
    sorted_dic_by_value_gt = sorted(gt_counter_per_class.items(), key=operator.itemgetter(0))
    # unpacking the list of tuples into two lists
    gtsorted_keys, gtsorted_values = zip(*sorted_dic_by_value_gt)
    # print(f"gtsorted_values:{gtsorted_values}")
    
    #遍历排序后的 ground truth 和检测计数，计算每个类别的假正例、真正例、类别名称、精度和召回率
    for i, (gtval,val) in enumerate(zip(gtsorted_values,sorted_values)):
        classname = gtsorted_keys[i]
        fp_val = fp_sorted[i]
        tp_val = tp_sorted[i]
        # print(f"fp_val:{fp_val}")
        # print(f"tp_val:{tp_val}")
        # print(f"gtval:{gtval}")    
        # print(f"classname:{classname}")

        # 计算每个类别的精度和召回率，并将它们转换为百分比形式的字符串
        precision = "{:.2f}%".format(round(tp_val/(tp_val+fp_val),4)*100)
        recall    = "{:.2f}%".format(round(tp_val/gtval,4)*100)
        # 将类别名称、精度和召回率添加到之前初始化的列表中
        categories.append(classname)
        precisions.append(precision)
        recalls.append(recall)    
    # print(f"categories:{categories}")
    # print(f"precisions:{precisions}")
    # print(f"recalls:{recalls}")
    #Create a figure and an axis
    ## 创建一个 Matplotlib 图形和坐标轴
    fig, ax = plt.subplots()

    # Hide actual axes，设置坐标轴的显示属性，使坐标轴紧凑并隐藏它们
    ax.axis('tight')
    ax.axis('off')

    # Table data: Include categories, precision, and recall in the table
    #准备表格的数据，包括表头和每个类别的名称、精度和召回率
    table_data = [
        ["Category", "Precision", "Recall"]
    ] + list(zip(categories, precisions, recalls))
    # print(f"table_data:{table_data}")
    
    # Create the table
    # 在坐标轴上创建一个表格，设置文本内容、列标签位置和表格位置
    table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')

    # Set table background color
    # 为表格单元格设置背景颜色，默认为浅蓝色
    cell_colors = [[(0.8, 0.8, 1)] * len(table_data[0]) for _ in range(len(table_data))]
    #如果当前行是表头行（即第一行），则设置其颜色为另一种蓝色
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            if i == 0:
                cell_colors[i][j] = (0.6, 0.8, 1)  # Header row color

    #设置表格的自动字体大小为关闭（手动设置），字体大小为 12，并缩放表格
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    # Set row height
    # 如果函数调用时指定了 row_height，则设置表格每一行的高度，并为表头行设置字体大小和加粗
    if row_height:
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                if (i, j) in table._cells:  # Check if cell exists
                    cell = table._cells[(i, j)]
                    cell.set_height(row_height)
                    if i == 0:  # Only set font properties for the header row
                        cell._text.set_fontsize(12)
                        cell._text.set_fontweight('bold')

    # Set column width
    # 如果函数调用时指定了 column_width，则设置表格每一列的宽度
    if column_width:
        for j in range(len(table_data[0])):
            for i in range(len(table_data)):
                if (i, j) in table._cells:  # Check if cell exists
                    cell = table._cells[(i, j)]
                    cell.set_width(column_width)
    # 为表格的每个单元格设置背景颜色
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(cell_colors[i][j])

    #如果提供了 save_path，则将图形保存为 PNG 格式的文件，并关闭图形；否则显示图形
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()



"""
 Draw plot using Matplotlib
 定义了 draw_plot_func 函数，使用 Matplotlib 绘制水平条形图，
 通常用于展示分类问题中各个类别的统计数据，如真正例（True Positives, TP）、假正例（False Positives, FP）和假负例（False Negatives, FN）
 接受多个参数，包括要绘制的数据字典、类别数量、窗口标题、图表标题、x轴标签、输出路径、是否显示图表、图表颜色和真正例的条形图数据
"""
def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    #对输入的字典按值进行排序，并将排序后的键（类别名）和值（数值）分别存储到两个列表中
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
     
    # 如果提供了 true_p_bar 参数，计算假正例和真正例的值，用于绘制多颜色的条形图
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - pink -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        # add legend，在图表的右下角添加图例
        plt.legend(loc='lower right')
        """
         Write number on side of bar，在条形图上绘制文本，显示数值
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)

    # 如果没有提供 true_p_bar，则只绘制单色（由 plot_color 指定）的条形图
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar，绘制条形上的文本，显示数值
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val) # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    # fig.canvas.set_window_title(window_title)设置显示图表时窗口的标题
    fig.canvas.manager.set_window_title(window_title)
    # write classes in y axis，设置y轴的刻度，显示排序后的类别名称
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly，调整图表的大小和布局，确保所有文本都可见
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height 
    top_margin = 0.15 # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot，将图表保存到指定的路径
    fig.savefig(output_path)
    # show image，参数为真，显示图表，则调用 Matplotlib 的 show 函数
    if to_show:
        plt.show()
    # close the plot
    plt.close()

# 计算两个框的阈值
def compute_iou(box1, box2):
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
    bbox1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    bbox2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (bbox1_area + bbox2_area - inter_area + 1e-6)
    return iou

"""
 Create a ".temp_files/" and "output/" directory
 这段代码负责创建临时文件目录和输出目录，如果这些目录已经存在，则重置
"""
# 定义一个变量 `TEMP_FILES_PATH` 用于存储临时文件路径
TEMP_FILES_PATH = ".temp_files"
# 检查 `TEMP_FILES_PATH` 指定的目录是否存在，如果不存在，使用 `os.makedirs` 创建它
if not os.path.exists(TEMP_FILES_PATH): # if it doesn't exist already
    os.makedirs(TEMP_FILES_PATH)
# 如果输出目录已存在，使用 `shutil.rmtree` 删除它，准备重置
if os.path.exists(output_files_path): # if it exist already
    # reset the output directory
    shutil.rmtree(output_files_path)
# 重新创建输出目录
os.makedirs(output_files_path)

# 如果变量 draw_plot 为真，则在输出目录下创建一个名为 classes 的子目录，用于存放按类别分类的图表
if draw_plot:
    os.makedirs(os.path.join(output_files_path, "classes"))
# 如果变量 show_animation 为真，则在输出目录下的 images 目录中创建一个名为 detections_one_by_one 的子目录，用于存放动画帧图像
if show_animation:
    os.makedirs(os.path.join(output_files_path, "images", "detections_one_by_one"))

#################3准备工作，加载和处理地面真实（ground-truth）数据
"""
 ground-truth
     Load each of the ground-truth files into a temporary ".json" file.
     Create a list of all the class names present in the ground-truth (gt_classes).
"""
# get a list with the ground-truth files
ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
if len(ground_truth_files_list) == 0:
    error("Error: No ground-truth files found!")
ground_truth_files_list.sort()
# dictionary with counter per class
gt_counter_per_class = {}
counter_images_per_class = {}

gt_files = []
for txt_file in ground_truth_files_list:
    #print(txt_file)
    file_id = txt_file.split(".txt", 1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    # check if there is a correspondent detection-results file
    temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
    if not os.path.exists(temp_path):
        error_msg = "Error. File not found: {}\n".format(temp_path)
        error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
        error(error_msg)
    lines_list = file_lines_to_list(txt_file)
    # create ground-truth dictionary
    bounding_boxes = []
    is_difficult = False
    already_seen_classes = []
    for line in lines_list:
        try:
            if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
            else:
                    class_name, left, top, right, bottom = line.split()
        except ValueError:
            error_msg = "Error: File " + txt_file + " in the wrong format.\n"
            error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
            error_msg += " Received: " + line
            error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
            error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
            error(error_msg)
        # check if class is in the ignore list, if yes skip
        if class_name in args.ignore:
            continue
        bbox = left + " " + top + " " + right + " " +bottom
        if is_difficult:
            bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
            is_difficult = False
        else:
            bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
            # count that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1

            if class_name not in already_seen_classes:
                if class_name in counter_images_per_class:
                    counter_images_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    counter_images_per_class[class_name] = 1
                already_seen_classes.append(class_name)


    # dump bounding_boxes into a ".json" file
    new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
    gt_files.append(new_temp_file)
    with open(new_temp_file, 'w') as outfile:
        json.dump(bounding_boxes, outfile)

gt_classes = list(gt_counter_per_class.keys())
# let's sort the classes alphabetically
gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)
print(gt_classes)
print(gt_counter_per_class)


################################################
"""
检查命令行参数 --set-class-iou 的格式和有效性的
 Check format of the flag --set-class-iou (if used)
    e.g. check if class exists
"""
if specific_iou_flagged:
    n_args = len(args.set_class_iou)
    error_msg = \
        '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
    if n_args % 2 != 0:
        error('Error, missing arguments. Flag usage:' + error_msg)
    # [class_1] [IoU_1] [class_2] [IoU_2]
    # specific_iou_classes = ['class_1', 'class_2']
    specific_iou_classes = args.set_class_iou[::2] # even
    # iou_list = ['IoU_1', 'IoU_2']
    iou_list = args.set_class_iou[1::2] # odd
    if len(specific_iou_classes) != len(iou_list):
        error('Error, missing arguments. Flag usage:' + error_msg)
    for tmp_class in specific_iou_classes:
        if tmp_class not in gt_classes:
                    error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
    for num in iou_list:
        if not is_float_between_0_and_1(num):
            error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

"""
加载检测结果文件，并将它们处理成统一的格式
 detection-results
     Load each of the detection-results files into a temporary ".json" file.
"""
# get a list with the detection-results files
# 获取检测结果的.txt文件，并排序
dr_files_list = glob.glob(DR_PATH + '/*.txt')
dr_files_list.sort()

# 初始化检测结果列表
detections = []

bounding_boxes = []  # 初始化一个空列表来存储当前类别的检测框
    # 遍历所有检测结果文件
for txt_file in dr_files_list:
    
    # file_id = txt_file.split(".txt", 1)[0]
    # file_id = os.path.basename(os.path.normpath(file_id))
    file_id = os.path.splitext(os.path.basename(txt_file))[0]
    lines = file_lines_to_list(txt_file) 

        # 遍历文件中的每一行
    for line in lines:
            # 解析每一行的数据
        try:
            parts = line.split()
            tmp_class_name = parts[0]
            confidence = parts[1]
            left, top, right, bottom = map(float, parts[2:])
        except ValueError:
            error_msg = "Error: File " + txt_file + " in the wrong format."
            error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>"
            error_msg += " Received: " + line
            error(error_msg)
            #  构建检测框的坐标列表
        bbox = [left, top, right, bottom]
        if tmp_class_name == "stain":  # 假设无标签的检测结果使用 "stain" 作为占位符
            bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox, "labeled": False})
        else:
            bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox, "labeled": True, "class_name": tmp_class_name})
    

    # 遍历文件中的每一行
    for line in lines:
        try:  ## 解析每一行的数据
            parts = line.split()
            tmp_class_name = parts[0]
            confidence = parts[1]
            left, top, right, bottom = map(float, parts[2:])
            bbox = [left, top, right, bottom]
            # 创建检测字典并加入file_id
            detection = {
                "class_name": tmp_class_name,
                "confidence": confidence,
                "bbox": bbox,
                "file_id": file_id  # 确保每个检测结果都有这个字段
            }
            detections.append(detection)
        except ValueError:
            error_msg = "Error: File " + txt_file + " in the wrong format."
            error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>"
            error_msg += " Received: " + line
            error(error_msg)

    # 保存更新后的检测结果
    for detection in detections:
        output_file_path = os.path.join(TEMP_FILES_PATH, detection["file_id"] + "_detections.json")
        with open(output_file_path, 'w') as outfile:
            json.dump([dict(detection)], outfile)       
####################################################################33
# 计算IoU的函数
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2
    inter_area = max(0, min(x2, x2_gt) - max(x1, x1_gt)) * max(0, min(y2, y2_gt) - max(y1, y1_gt))
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area != 0 else 0

# 读取真实框文件并返回一个字典，其中键是文件ID，值是包含所有真实框信息的列表
def read_ground_truth_files(ground_truth_files_list):
    ground_truth_data = defaultdict(list)
    for txt_file in ground_truth_files_list:
        file_id = os.path.splitext(os.path.basename(txt_file))[0]
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_name, left, top, right, bottom = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            bbox = [left, top, right, bottom]
            ground_truth_data[file_id].append({'class_name': class_name, 'bbox': bbox})
    return ground_truth_data

# 读取检测结果文件并返回一个字典，其中键是文件ID，值是包含所有检测框信息的列表
def read_detection_files(dr_files_list):
    detection_data = defaultdict(list)
    for txt_file in dr_files_list:
        file_id = os.path.splitext(os.path.basename(txt_file))[0]
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            tmp_class_name, confidence, left, top, right, bottom = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            bbox = [left, top, right, bottom]
            detection_data[file_id].append({'confidence': confidence, 'bbox': bbox, 'labeled': tmp_class_name != "stain"})
    return detection_data

# 匹配检测框和真实框
def match_detections_to_ground_truth(detection_data, ground_truth_data, iou_threshold=0.25):
    for file_id, detections in detection_data.items():
        for detection in detections:
            if not detection['labeled']:  # 只处理未标记的检测框
                max_iou = 0
                matched_gt = None
                for gt in ground_truth_data[file_id]:
                    iou = compute_iou(detection['bbox'], gt['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        matched_gt = gt
                if max_iou >= iou_threshold:
                    detection['class_name'] = matched_gt['class_name']
                    detection['labeled'] = True
                else:
                    detection['labeled'] = False

########################################################################

ground_truth_data = read_ground_truth_files(ground_truth_files_list)
detection_data = read_detection_files(dr_files_list)

match_detections_to_ground_truth(detection_data, ground_truth_data)

# # 保存更新后的检测结果
# for file_id, detections in detection_data.items():
#     output_file_path = os.path.join(TEMP_FILES_PATH, file_id + "_detections.json")
#     with open(output_file_path, 'w') as outfile:
#         json.dump(detections, outfile)


       
"""
核心：计算每个类别的平均精度（Average Precision, AP）以及日志平均错误率（log-average miss rate, LAMR）。
 Calculate the AP for each class
"""
sum_AP = 0.0
ap_dictionary = {}
lamr_dictionary = {}
# open file to store the output
with open(output_files_path + "/output.txt", 'w') as output_file:
    output_file.write("# AP and precision/recall per class\n")
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        """
         Load detection-results of that class
        """
        dr_file = TEMP_FILES_PATH + "/" + file_id + "_detections.json"
        dr_data = json.load(open(dr_file))

        """
         Assign detection-results to ground-truth objects
        """
        nd = len(dr_data)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        # for idx, detection in enumerate(dr_data):
        #     file_id = detection["file_id"]

        # for detection in dr_data:
        for idx, detection in enumerate(dr_data):
            if 'file_id' not in detection:
                print(f"Missing 'file_id' in detection data: {detection}")
            else:
                file_id = detection["file_id"]

            if show_animation:
                # find ground truth image
                ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                #tifCounter = len(glob.glob1(myPath,"*.tif"))
                if len(ground_truth_img) == 0:
                    error("Error. Image not found with id: " + file_id)
                elif len(ground_truth_img) > 1:
                    error("Error. Multiple image with id: " + file_id)
                else: # found image
                    #print(IMG_PATH + "/" + ground_truth_img[0])
                    # Load image
                    img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])
                    # load image with draws of multiple detections
                    img_cumulative_path = output_files_path + "/images/" + ground_truth_img[0]
                    if os.path.isfile(img_cumulative_path):
                        img_cumulative = cv2.imread(img_cumulative_path)
                    else:
                        img_cumulative = img.copy()
                    # Add bottom border to image
                    bottom_border = 60
                    BLACK = [0, 0, 0]
                    img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
            # assign detection-results to ground truth object if any
            # open ground-truth with that file_id
            gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            # bb = [ float(x) for x in detection["bbox"].split() ]
            bb = [ float(x) for x in detection["bbox"] ]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [ float(x) for x in obj["bbox"].split() ]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            if show_animation:
                status = "NO MATCH FOUND!" # status is only used in the animation
            # set minimum overlap
            min_overlap = MINOVERLAP
            if specific_iou_flagged:
                if class_name in specific_iou_classes:
                    index = specific_iou_classes.index(class_name)
                    min_overlap = float(iou_list[index])

            if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, 'w') as f:
                                    f.write(json.dumps(ground_truth_data))
                            if show_animation:
                                status = "MATCH!"
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
                            if show_animation:
                                status = "REPEATED MATCH!"

            else:
                # false positive
                fp[idx] = 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"
        
                

            """
             Draw image to show animation
            """
            if show_animation:
                height, widht = img.shape[:2]
                # colors (OpenCV works with BGR)
                white = (255,255,255)
                light_blue = (255,200,100)
                green = (0,255,0)
                light_red = (30,30,255)
                # 1st line
                margin = 10
                v_pos = int(height - margin - (bottom_border / 2.0))
                text = "Image: " + ground_truth_img[0] + " "
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
                if ovmax != -1:
                    color = light_red
                    if status == "INSUFFICIENT OVERLAP":
                        text = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100)
                    else:
                        text = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(min_overlap*100)
                        color = green
                    img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                # 2nd line
                v_pos += int(bottom_border / 2.0)
                rank_pos = str(idx+1) # rank position (idx starts at 0)
                text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(float(detection["confidence"])*100)
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                color = light_red
                if status == "MATCH!":
                    color = green
                text = "Result: " + status + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                font = cv2.FONT_HERSHEY_SIMPLEX
                if ovmax > 0: # if there is intersections between the bounding-boxes
                    bbgt = [ int(round(float(x))) for x in gt_match["bbox"].split() ]
                    cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                    cv2.rectangle(img_cumulative,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                    cv2.putText(img_cumulative, class_name, (bbgt[0],bbgt[1] - 5), font, 0.6, light_blue, 1, cv2.LINE_AA)
                bb = [int(i) for i in bb]
                cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                cv2.rectangle(img_cumulative,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                cv2.putText(img_cumulative, class_name, (bb[0],bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                # # show image
                # cv2.imshow("Animation", img)
                # cv2.waitKey(20) # show for 20 ms
                # save image to output
                output_img_path = output_files_path + "/images/detections_one_by_one/" + class_name + "_detection" + str(idx) + ".jpg"
                cv2.imwrite(output_img_path, img)
                # save the image with all the objects drawn to it
                cv2.imwrite(img_cumulative_path, img_cumulative)

        #print(tp)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        #print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        #print(prec)

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)
        """
         Write to output.txt
        """
        rounded_prec = [ '%.2f' % elem for elem in prec ]
        rounded_rec = [ '%.2f' % elem for elem in rec ]
        output_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
        if not args.quiet:
            print(text)
        ap_dictionary[class_name] = ap

        n_images = counter_images_per_class[class_name]
        lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
        lamr_dictionary[class_name] = lamr

        """
         Draw plot
        """
        if draw_plot:
            plt.plot(rec, prec, '-o')
            # add a new penultimate point to the list (mrec[-2], 0.0)
            # since the last line segment (and respective area) do not affect the AP value
            area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
            area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
            plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
            # set window title
            fig = plt.gcf() # gcf - get current figure
            # fig.canvas.set_window_title('AP ' + class_name)
            fig.canvas.manager.set_window_title('AP ' + class_name)
            # set plot title
            plt.title('class: ' + text)
            #plt.suptitle('This is a somewhat long figure title', fontsize=16)
            # set axis titles
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # optional - set axes
            axes = plt.gca() # gca - get current axes
            axes.set_xlim([0.0,1.0])
            axes.set_ylim([0.0,1.05]) # .05 to give some extra space
            # Alternative option -> wait for button to be pressed
            #while not plt.waitforbuttonpress(): pass # wait for key display
            # Alternative option -> normal display
            #plt.show()
            # save the plot
            fig.savefig(output_files_path + "/classes/" + class_name + ".png")
            plt.cla() # clear axes for next plot

    if show_animation:
        cv2.destroyAllWindows()

    output_file.write("\n# mAP of all classes\n")
    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP*100)
    output_file.write(text + "\n")
    print(text)


"""
 Draw false negatives
"""
if show_animation:
    pink = (203,192,255)
    for tmp_file in gt_files:
        ground_truth_data = json.load(open(tmp_file))
        #print(ground_truth_data)
        # get name of corresponding image
        start = TEMP_FILES_PATH + '/'
        img_id = tmp_file[tmp_file.find(start)+len(start):tmp_file.rfind('_ground_truth.json')]
        img_cumulative_path = output_files_path + "/images/" + img_id + ".jpg"
        img = cv2.imread(img_cumulative_path)
        if img is None:
            img_path = IMG_PATH + '/' + img_id + ".jpg"
            img = cv2.imread(img_path)
        # draw false negatives
        for obj in ground_truth_data:
            if not obj['used']:
                bbgt = [ int(round(float(x))) for x in obj["bbox"].split() ]
                cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),pink,2)
        cv2.imwrite(img_cumulative_path, img)

# remove the temp_files directory
shutil.rmtree(TEMP_FILES_PATH)

"""
 Count total of detection-results
"""
# iterate through all the files
det_counter_per_class = {}
for txt_file in dr_files_list:
    # get lines to list
    lines_list = file_lines_to_list(txt_file)
    for line in lines_list:
        class_name = line.split()[0]
        # check if class is in the ignore list, if yes skip
        if class_name in args.ignore:
            continue
        # count that object
        if class_name in det_counter_per_class:
            det_counter_per_class[class_name] += 1
        else:
            # if class didn't exist yet
            det_counter_per_class[class_name] = 1
#print(det_counter_per_class)
dr_classes = list(det_counter_per_class.keys())


"""
 Plot the total number of occurences of each class in the ground-truth
"""
if draw_plot:
    window_title = "ground-truth-info"
    plot_title = "ground-truth\n"
    plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
    x_label = "Number of objects per class"
    output_path = output_files_path + "/ground-truth-info.png"
    to_show = False
    plot_color = 'forestgreen'
    draw_plot_func(
        gt_counter_per_class,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        '',
        )

"""
 Write number of ground-truth objects per class to results.txt
"""
with open(output_files_path + "/output.txt", 'a') as output_file:
    output_file.write("\n# Number of ground-truth objects per class\n")
    for class_name in sorted(gt_counter_per_class):
        output_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

"""
 Finish counting true positives
"""
for class_name in dr_classes:
    # if class exists in detection-result but not in ground-truth then there are no true positives in that class
    if class_name not in gt_classes:
        count_true_positives[class_name] = 0
#print(count_true_positives)

"""
 Plot the total number of occurences of each class in the "detection-results" folder
"""
if draw_plot:
    window_title = "detection-results-info"
    # Plot title
    plot_title = "detection-results\n"
    plot_title += "(" + str(len(dr_files_list)) + " files and "
    count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
    plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
    # end Plot title
    x_label = "Number of objects per class"
    output_path = output_files_path + "/detection-results-info.png"
    to_show = False
    plot_color = 'forestgreen'
    true_p_bar = count_true_positives
    draw_plot_func(
        det_counter_per_class,
        len(det_counter_per_class),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        true_p_bar
        )


'''
draw precision and racall stable
'''
output_path = output_files_path + "/precision-recall-info.png"
create_precision_recall_table(gt_counter_per_class,det_counter_per_class,n_classes,save_path=output_path);


"""
 Write number of detected objects per class to output.txt
"""
with open(output_files_path + "/output.txt", 'a') as output_file:
    output_file.write("\n# Number of detected objects per class\n")
    for class_name in sorted(dr_classes):
        n_det = det_counter_per_class[class_name]
        text = class_name + ": " + str(n_det)
        text += " (tp:" + str(count_true_positives[class_name]) + ""
        text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
        output_file.write(text)

"""
 Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
"""
if draw_plot:
    window_title = "lamr"
    plot_title = "log-average miss rate"
    x_label = "log-average miss rate"
    output_path = output_files_path + "/lamr.png"
    to_show = False
    plot_color = 'royalblue'
    draw_plot_func(
        lamr_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
        )

"""
 Draw mAP plot (Show AP's of all classes in decreasing order)
"""
if draw_plot:
    window_title = "mAP"
    plot_title = "mAP = {0:.2f}%".format(mAP*100)
    x_label = "Average Precision"
    output_path = output_files_path + "/mAP.png"
    to_show = False
    plot_color = 'royalblue'
    draw_plot_func(
        ap_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
        )

