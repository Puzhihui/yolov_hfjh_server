import os
from flask import Flask, jsonify, request
import json
import time
import datetime
import math

import cv2

from logserver import LogServer
from ultralytics import YOLO


def get_str_datetime():
    return str(datetime.datetime.now().year) + '_' + str(datetime.datetime.now().month) + '_' + str(datetime.datetime.now().day) + '_' + str(datetime.datetime.now().hour)


global log_server

log_server = LogServer(app='adc-service-hfjh', log_path="./log")
filename = get_str_datetime()
log_server.re_configure_logging('eagle_' + str(filename) + "_log.txt")
print("logfile is: ", str(filename) + "_log.txt")
log_server.logging("logfile is %s" % (str(filename) + "_log.txt"))
print("**********load log config file success*******************")

app = Flask(__name__)

model_path = './runs/detect/train6/weights/best.pt'
log_server.logging("loading yolov8 model, model path:{}".format(model_path))
model = YOLO(model_path)
log_server.logging("model load sucess!!!")
print("model load sucess!!! model path:{}".format(model_path))

# 切割小图参数
row_split = [946, 1650]
# col_split = [0, width]
little_img_size = [704, 704]  # 小图尺寸
split_overlap_size = [50, 50]  # 切小图时重叠的尺寸
confidence_thres = 0.5
width_thres = 150  # 预测的缺陷宽度，大于阈值则过滤
height_thres = 150  # 预测的缺陷高度，大于阈值则过滤
is_save_img = True


def get_patch_img(image_cv, patch_size, overlap_size=[50, 50]):
    height, width = image_cv.shape[:2]
    overlapW, overlapH = overlap_size
    if patch_size[0] >= width:
        overlapW = 0
    if patch_size[1] >= height:
        overlapH = 0
    patch_w_num = math.ceil(width  / (patch_size[0]-overlapW))
    patch_h_num = math.ceil(height / (patch_size[1]-overlapH))
    patch_image_dict = dict()
    x1, y1, x2, y2 = 0, 0, patch_size[0], patch_size[1]
    for i in range(patch_h_num):  # y方向
        for j in range(patch_w_num):  # x方向
            cropped_image = image_cv[y1:y2, x1:x2]
            if i not in patch_image_dict.keys():
                patch_image_dict[i] = []
            patch_image_dict[i].append({"img": cropped_image, "points": [x1, y1, x2, y2]})

            over_pixel_x = (x2 + patch_size[0] - overlapW) - width  # 判断有边界是否大于图片宽度
            if over_pixel_x > 0:
                slide_step_x = patch_size[0] - over_pixel_x
            else:
                slide_step_x = patch_size[0] - overlapW
            x1, x2 = slide_step_x + x1, slide_step_x + x2

        x1, x2 = 0, patch_size[0]
        over_pixel_y = (y2 + patch_size[1]) - height  # 这里没有- overlapH，是因为高度本来就是704，不需要再滑动
        if over_pixel_y > 0:
            slide_step_y = patch_size[1] - over_pixel_y
        else:
            slide_step_y = patch_size[1] - overlapH
        y1, y2 = slide_step_y + y1, slide_step_y + y2

    return patch_image_dict


def crop_img(image_path, image, row, col):
    split_imgs = dict()
    start_row, end_row = row
    start_col, end_col = col
    cropped_image = image[start_row:end_row, start_col:end_col]
    patch_image_dict = get_patch_img(cropped_image, little_img_size, overlap_size=split_overlap_size)
    basename = os.path.basename(image_path)
    name_without_extension = os.path.splitext(basename)[0]
    for R, patch_imgs in patch_image_dict.items():
        split_imgs[R] = dict()
        for C, path_img in enumerate(patch_imgs):
            split_imgs[R][C] = path_img
            # 保存格式 recipe@lot@layer@TopFlat@basename@splitsize@Rn@Cn.Bmp
            if is_save_img:
                dst_basename = '{}@{}@{}@{}@{}@{}@R{}C{}.Bmp'.format('recipe3', 'lot3', 'layer0', 'TopFlat', '704', name_without_extension, R, C)
                cv2.imwrite(os.path.join(r"D:\Solution\datas\HFJH\AAI0005JM1\#19\Layer_0\TopFlat\split_img", dst_basename), path_img["img"])
    return split_imgs


def split_img(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    col = [0, width]
    row = row_split
    split_imgs = crop_img(image_path, image, row, col)
    return split_imgs, row, col


def parse_input(receiveData):
    input_dict = json.loads(receiveData)
    img_id = input_dict.get("ImageID")
    recipe_name = input_dict.get("RecipeName")
    lot_id = input_dict.get("LotID")
    wafer_id = input_dict.get("WaferID")
    img_path = input_dict.get("ImagePath")
    return img_id, recipe_name, lot_id, wafer_id, img_path


def cal_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # 计算并集面积
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    union_area = area_a + area_b - intersection_area
    # 计算IOU
    iou = intersection_area / union_area

    surround = False
    if a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]:
        surround = True
    if a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]:
        surround = True
    return iou, surround


def filter_boxes(ImageId, defect_dict):
    del_dict = dict()
    for row, col_dict in defect_dict.items():
        for col, pred_list in col_dict.items():
            if col + 1 in defect_dict[row].keys():
                next_col_prd_list = defect_dict[row][col + 1]
                for pred in pred_list:
                    box_pred = [pred[1]-pred[3]/2, pred[2]-pred[4]/2, pred[1]+pred[3]/2, pred[2]+pred[4]/2]
                    for next_pred in next_col_prd_list:
                        next_box_pred = [next_pred[1]-next_pred[3]/2, next_pred[2]-next_pred[4]/2, next_pred[1]+next_pred[3]/2, next_pred[2]+next_pred[4]/2]
                        iou, surround = cal_iou(box_pred, next_box_pred)
                        if iou > 0.5 or surround:
                            if pred[3] * pred[4] > next_pred[3] * next_pred[4]:
                                if 'R{}C{}'.format(row, col+1) not in del_dict.keys():
                                    del_dict['R{}C{}'.format(row, col+1)] = []
                                del_dict['R{}C{}'.format(row, col+1)].append(next_pred)
                            else:
                                if 'R{}C{}'.format(row, col) not in del_dict.keys():
                                    del_dict['R{}C{}'.format(row, col)] = []
                                del_dict['R{}C{}'.format(row, col)].append(pred)
    AlgoDefectData = []
    DefectId = 0
    box_list = []
    for row, col_dict in defect_dict.items():
        for col, pred_list in col_dict.items():
            pred_RC = 'R{}C{}'.format(row, col)
            if pred_RC in del_dict.keys():
                del_pred_list = del_dict[pred_RC]
                for pred in pred_list:
                    for del_pred in del_pred_list:
                        if pred == del_pred:
                            continue
                        else:
                            label, center_x, center_y, w, h, conf = pred
                            AlgoDefectData.append(
                                {
                                    "ImageID": ImageId,
                                    "CenterX": center_x,
                                    "CenterY": center_y,
                                    "Area": w*h,
                                    "Length": h,
                                    "Width": w,
                                    "Category": label,
                                    "Confidence": conf
                                }
                            )
                            DefectId += 1
                            box_list.append([label, [center_x - w / 2, center_y - h / 2, center_x + w / 2, center_y + h / 2], conf])
            else:
                for pred in pred_list:
                    label, center_x, center_y, w, h, conf = pred
                    AlgoDefectData.append(
                        {
                            "ImageID": ImageId,
                            "CenterX": center_x,
                            "CenterY": center_y,
                            "Area": w * h,
                            "Length": h,
                            "Width": w,
                            "Category": label,
                            "Confidence": conf
                        }
                    )
                    DefectId += 1
                    box_list.append([label, [center_x - w / 2, center_y - h / 2, center_x + w / 2, center_y + h / 2], conf])
    return AlgoDefectData, box_list, len(del_dict)


def draw_on_original_image(image_path, box_list, label=True, color=(0, 0, 255)):
    im0 = cv2.imread(image_path)  # BGR
    for pred in box_list:
        box = pred[1]
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(im0, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite("/data3/pzh/data/hfjh/test_server/vis.bmp", im0)


def merge_pred(img_id, split_imgs, pred_dict, split_row, split_col):
    defect_dict = dict()
    for row, col_dict in pred_dict.items():
        if row not in defect_dict.keys():
            defect_dict[row] = dict()
        for col, pred_list in col_dict.items():
            defect_dict[row][col] = []
            # 获取该crop img的起始坐标
            if row in split_imgs.keys() and col in split_imgs[row].keys():
                points = split_imgs[row][col]["points"]
                start_x, start_y = points[0], points[1]
            for pred in pred_list:
                label, center_x, center_y, w, h, conf = pred
                center_or_x = start_x + center_x + split_col[0]
                center_or_y = start_y + center_y + split_row[0]
                # center_or_x = center_x + split_col[0] + (little_img_size[0] - split_overlap_size[0]) * col
                # center_or_y = center_y + split_row[0] + (little_img_size[1] - split_overlap_size[1]) * row
                defect_dict[row][col].append([label, center_or_x, center_or_y, w, h, conf])
    AlgoDefectData, box_list, del_num = filter_boxes(img_id, defect_dict)
    return AlgoDefectData, box_list, del_num


@app.route('/ADC/', methods=['POST'])
def run_adc():
    service_time_start = time.time()
    log_server.logging("================================start=============================")
    AlgoDefectData = []
    if request.method == 'POST':
        img_id, recipe_name, lot_id, wafer_id, img_path = parse_input(request.data.decode('utf-8'))
        if not img_path or not os.path.exists(img_path):
            return json.dumps({"errorcode": 1, "msg": "ImagePath is not exists: {}".format(img_path), "AlgoDefectData": []})
        img_path = img_path.replace('\\', '/')
        log_server.logging("img path: {}".format(img_path))
        split_imgs, row_s, col_s = split_img(img_path)
        pred_dict = dict()
        for row, col_values in split_imgs.items():
            input_img_list = [value["img"] for value in col_values.values()]
            results = model(input_img_list)
            pred_dict[row] = dict()
            for col, result in enumerate(results):
                boxes = result.boxes
                xywh_list = boxes.xywh.cpu().tolist()
                if len(xywh_list) == 0:
                    continue
                index2class = result.names
                cls_list = boxes.cls.cpu().tolist()
                conf_list = boxes.conf.cpu().tolist()
                # pred_dict[row][col] = []
                for xywh, cls, conf in zip(xywh_list, cls_list, conf_list):
                    if conf < confidence_thres:
                        continue
                    center_x, center_y, w, h = xywh
                    if w > width_thres or h > height_thres:
                        continue
                    if col not in pred_dict[row].keys():
                        pred_dict[row][col] = []
                    pred_dict[row][col].append([index2class[cls], center_x, center_y, w, h, conf])
        log_server.logging("detect {} defect from crop img".format(sum(len(sublist) for subdict in pred_dict.values() for sublist in subdict.values())))
        AlgoDefectData, box_list, del_num = merge_pred(img_id, split_imgs, pred_dict, row_s, col_s)
        log_server.logging("after merge pred, del {} defect".format(del_num))
    all_time = time.time() - service_time_start
    log_server.logging("detect all time is {:.4f}s".format(all_time))
    log_server.logging("================================end=============================")
    return json.dumps({"errorcode": 0,
                       "msg": "success",
                       "AlgoDefectData": AlgoDefectData})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3079, debug=False, threaded=True)
