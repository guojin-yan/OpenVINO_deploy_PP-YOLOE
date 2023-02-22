import cv2 as cv
from openvino_predictor import Predictor
from process import process_image,process_result,draw_box,read_lable

def yoloe_infer():
    '''-------------------1. 导入相关信息 ----------------------'''
    # yoloe_model_path = "E:/Text_Model/pp-yoloe/ppyoloe_plus_crn_s_80e_coco.onnx"
    yoloe_model_path = "E:/Text_Model/pp-yoloe/ppyoloe_plus_crn_s_80e_coco.xml"
    image_path = "E:/Text_dataset/YOLOv5/0001.jpg"
    lable_path = "E:/Git_space/基于OpenVINO部署PP-YOLOE模型/model/lable.txt";
    '''-------------------2. 创建模型预测器 ----------------------'''
    predictor = Predictor(model_path = yoloe_model_path)
    '''-------------------3. 预处理模型输入数据 ----------------------'''
    image = cv.imread(image_path)
    input_image = process_image(image, 640)
    '''-------------------4. 模型推理 ----------------------'''
    results = predictor.predict(input_data=input_image)
    '''-------------------5. 后处理预测结果 ----------------------'''
    boxes_name = predictor.get_outputs_name(0)
    conf_name = predictor.get_outputs_name(1)
    
    boxes, scores, classes = process_result(box_results=results[boxes_name], conf_results=results[conf_name]) # 处理结果
    lables = read_lable(lable_path=lable_path) # 读取lable
    result_image = draw_box(image=image, boxes=boxes, scores=scores, classes=classes, lables=lables) # 绘制结果
    cv.imshow("result",result_image)
    cv.waitKey(0)



if __name__ == '__main__':
    yoloe_infer()
    