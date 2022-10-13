import cv2 as cv
import numpy as np
import tensorflow as tf


def process_image(input_image, size):
    """输入图片与处理方法，按照PP-Yoloe模型要求预处理图片数据

    Args:
        input_image (uint8): 输入图片矩阵
        size (int): 模型输入大小

    Returns:
        float32: 返回处理后的图片矩阵数据
    """
    max_len = max(input_image.shape)
    img = np.zeros([max_len,max_len,3],np.uint8)
    img[0:input_image.shape[0],0:input_image.shape[1]] = input_image # 将图片放到正方形背景中
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)  # BGR转RGB
    img = cv.resize(img, (size, size), cv.INTER_NEAREST) # 缩放图片
    img = np.transpose(img,[2, 0, 1]) # 转换格式
    img = img / 255.0 # 归一化
    img = np.expand_dims(img,0) # 增加维度
    return img


def process_result(box_results, conf_results):
    """按照PP-Yolove模型输出要求，处理数据，非极大值抑制，提取预测结果

    Args:
        box_results (float32): 预测框预测结果
        conf_results (float32): 置信度预测结果
    Returns:
        float: 预测框
        float: 分数
        int: 类别
    """
    conf_results = np.transpose(conf_results,[0, 2, 1]) # 转换数据通道
    # 设置输出形状
    box_results =box_results.reshape(8400,4) 
    conf_results = conf_results.reshape(8400,80)
    scores = []
    classes = []
    boxes = []
    for i in range(8400):
        conf = conf_results[i,:] # 预测分数
        score = np.max(conf) # 获取类别
        # 筛选较小的预测类别
        if score > 0.5:
            classes.append(np.argmax(conf)) 
            scores.append(score) 
            boxes.append(box_results[i,:])
    scores = np.array(scores)
    boxes = np.array(boxes)
    # 非极大值抑制筛选重复的预测结果
    indexs = tf.image.non_max_suppression(boxes,scores,len(scores),0.25,0.35)
    # 处理非极大值抑制后的结果
    result_box = []
    result_score = []
    result_class = []
    for i, index in enumerate(indexs):
        result_score.append(scores[index])
        result_box.append(boxes[index,:])
        result_class.append(classes[index])
    # 返沪结果转为矩阵
    return np.array(result_box),np.array(result_score),np.array(result_class)
        
def draw_box(image, boxes, scores, classes, lables):
    """将预测结果绘制到图像上

    Args:
        image (uint8): 原图片
        boxes (float32): 预测框
        scores (float32): 分数
        classes (int): 类别
        lables (str): 标签

    Returns:
        uint8: 标注好的图片
    """
    scale = max(image.shape) / 640.0 # 缩放比例
    for i in range(len(classes)):
        box = boxes[i,:]

        x1 = int(box[0] * scale)
        y1 = int(box[1] * scale)
        x2 = int(box[2] * scale)
        y2 = int(box[3] * scale)
        
        lable = lables[classes[i]]
        score = scores[i]
        cv.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2, cv.LINE_8)
        cv.putText(image,lable+":"+str(score),(x1,y1-10),cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        
    return image

def read_lable(lable_path):
    """读取lable文件

    Args:
        lable_path (str): 文件路径

    Returns:
        str: _description_
    """
    f = open(lable_path)
    lable = [] 
    line = f.readline()
    while line:
        lable.append(line)
        line = f.readline() 
    f.close()
    return lable
          
      
        


