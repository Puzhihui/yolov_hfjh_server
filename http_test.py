import glob
import os
import requests
import json
import cv2


def draw_on_original_image(image_path, vis_save_path, im0, box_list, label=True, color=(0, 0, 255)):

    for i, pred in enumerate(box_list):
        box = pred[1]
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.putText(im0, str(len(box_list)), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 255, 255), 9)
        cv2.rectangle(im0, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(im0, str(i+1), (p2[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(vis_save_path, "vis_{}".format(os.path.basename(image_path))), im0)


url = "http://169.254.136.57:3079/ADC/"
# detect_path = r"D:\Solution\datas\HFJH\AAI0005JM1\#19\Layer_0\TopFlat"
detect_path = "D:\Solution\datas\HFJH\AAI0005JM1\#3\layer0\TopFlat"

if os.path.isdir(detect_path):
    img_path_list = glob.glob(os.path.join(detect_path, "*.Bmp"))
elif os.path.isfile(detect_path):
    img_path_list = [detect_path]
for img_path in img_path_list:
    resquest_data = json.dumps({"ImageId": 1,
                                "RecipeName": "1111",
                                "LotID": "2222",
                                "WaferID": "3333",
                                # "ImagePath": "/data3/pzh/data/hfjh/test_original_img/F019.Bmp"
                                "ImagePath": img_path
                                })

    response = requests.post(url, data=resquest_data)
    results = response.text
    results = json.loads(results)
    if results["errorcode"] != 0:
        print("server error, errorcode:{}, msg: {}".format(results["errorcode"], results["msg"]))
    else:
        data_list = results["AlgoDefectData"]
        box_list = []
        im0 = cv2.imread(img_path)  # BGR
        height, width = im0.shape[:2]
        for data in data_list:
            confidence = data["Confidence"]
            label = data["Category"]
            center_x = data["CenterX"]
            center_y = data["CenterY"]
            w = data["Width"]
            h = data["Length"]

            orignal_x1 = center_x - w / 2
            orignal_y1 = center_y - h / 2
            orignal_x2 = center_x + w / 2
            orignal_y2 = center_y + h / 2
            box_list.append([label, [orignal_x1, orignal_y1, orignal_x2, orignal_y2], confidence])
            if orignal_x1 > width or orignal_x2 >width or orignal_y1 > height or orignal_y2 > height:
                raise "point out of range"
            if orignal_x1 < 0 or orignal_x2 < 0 or orignal_y1 < 0 or orignal_y2 < 0:
                raise "point is negative number"
        draw_on_original_image(img_path, os.path.dirname(img_path), im0, box_list)
    print("success: {}".format(img_path))