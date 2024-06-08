# aidlux相关
import json
import re
import sys
import android

droid = android.Android()

from cvs import *
import aidlite_gpu
from utils import detect_postprocess, preprocess_img, draw_detect_res

# 七牛云相关
from qiniu import Auth, put_file
from qiniu import CdnManager
import time
import requests
import cv2
import datetime

# 配置七牛云信息
access_key = ""
secret_key = ""
bucket_name = ""
bucket_url = ""
q = Auth(access_key, secret_key)
cdn_manager = CdnManager(q)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.47'}

# 加载模型
model_path = 'yolov5s-fp16.tflite'
# 定义输入输出shape
in_shape = [1 * 640 * 640 * 3 * 4]
out_shape = [1 * 25200 * 7 * 4, 1 * 3 * 80 * 80 * 7 * 4, 1 * 3 * 40 * 40 * 7 * 4, 1 * 3 * 20 * 20 * 7 * 4]
# 载入模型
aidlite = aidlite_gpu.aidlite()
# 载入yolov5检测模型
aidlite.ANNModel(model_path, in_shape, out_shape, 4, 0)


# 将本地图片上传到七牛云中
def upload_img(file_name, file_path):
    # generate token
    token = q.upload_token(bucket_name, file_name)
    put_file(token, file_name, file_path)


# 获得七牛云服务器上file_name的图片外链
def get_img_url(file_name):
    img_url = 'http://%s/%s' % (bucket_url, file_name)
    return img_url


def get_valid_id():
    while True:
        res = droid.getInput("喵码是用于发布警报的标识码 可在微信公众号喵提醒获取", "请输入喵码")
        input_id = res.result
        if len(input_id) == 7 and input_id.isalnum() and valid_id(input_id):
            return input_id
        droid.makeToast("喵码错误 请重新输入")


def valid_id(input_id):
    response = requests.post(
        "http://miaotixing.com/trigger?id=" + input_id + "&text=尝试连接" + "&ts=" + str(time.time()) + "&type=json",
        headers=headers)
    try:
        data = response.json()
        if data['code'] == 0:
            return True
        else:
            return False
    except ValueError:
        return False


def send_notification(name, id):
    # 取出文件名为name的图片的url
    url_receive = get_img_url(name)
    text = "告警图片:" + url_receive
    ts = str(time.time())  # 时间戳
    request_url = "http://miaotixing.com/trigger?"
    requests.post(request_url + "id=" + id + "&text=" + text + "&ts=" + ts + "&type=json", headers=headers)


# 输入喵码
miao_id = get_valid_id()
# 初始化标记
person = 0
fire = 0
personAlert = 0
fireAlert = 0
# 用户要求对person警报(家中有人时可置False 只警报fire)
personTurnOn = True
alert_interval = 30
# 启动摄像头
cap = cvs.VideoCapture(0)
# 初始化报警时间戳
last_alert_time = time.time() - alert_interval

while True:
    frame = cap.read()
    if frame is None:
        continue

    # 预处理
    img = preprocess_img(frame, target_shape=(640, 640), div_num=255, means=None, stds=None)
    aidlite.setInput_Float32(img, 640, 640)
    # 推理
    aidlite.invoke()
    pred = aidlite.getOutput_Float32(0)
    pred = pred.reshape(1, 25200, 7)[0]
    pred = detect_postprocess(pred, frame.shape, [640, 640, 3], conf_thres=0.5, iou_thres=0.45)
    res_img, person, fire = draw_detect_res(frame, pred, person, fire)

    # 如果识别到了person或fire 截取图片并保存在本地 待冷却时间结束后再警报
    if person == 1 or fire == 1:
        cv2.imwrite("detect_image.jpg", res_img)
        if person == 1:
            # 及时撤销标记
            person = 0
            personAlert = 1

        if fire == 1:
            # 及时撤销标记
            fire = 0
            fireAlert = 1

    # 如果[fire标记为1]或[person标记为1且用户要求对person警报] 且[冷却时间结束] 则发出警报
    if ((personAlert == 1 and personTurnOn) or fireAlert == 1) and time.time() - last_alert_time >= alert_interval:
        # 屏蔽时间内 不进行语音警报
        if 0 <= datetime.datetime.now().hour < 24:
            if personAlert == 1 and fireAlert == 1:
                droid.vibrate(1000)
                droid.ttsSpeak('警报 有人闯入且有火灾隐患')
            elif personAlert == 1 :
                droid.vibrate(1000)
                droid.ttsSpeak('警报 有人闯入')
            elif fireAlert == 1:
                droid.vibrate(1000)
                droid.ttsSpeak('警报 有火灾隐患')

        # 及时清空Alert 避免重复报
        personAlert = 0
        fireAlert = 0

        # 需要上传到七牛云上面的图片的路径
        image_up_name = "detect_image.jpg"
        # 获取当前时间
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # 更新报警时间戳
        last_alert_time = time.time()
        # 构建保存到七牛云上的图片名称
        image_qiniu_name = f"detect_image_{current_time}_{miao_id}.jpg"
        # 将图片上传到七牛云,并保存成image_qiniu_name的名称
        upload_img(image_qiniu_name, image_up_name)
        # 向喵提醒发出通知
        send_notification(image_qiniu_name, miao_id)

    cvs.imshow(res_img)
