# OpenVINO<sup>TM</sup>部署PaddlePaddle YOLOE模型—Python

# 1. 环境安装

&emsp;OpenVINO<sup>TM</sup>工具套件2022.1版于2022年3月22日正式发布，与以往版本相比发生了重大革新，提供预处理API函数、ONNX前端API、AUTO 设备插件，并且支持直接读入飞桨模型，在推理中中支持动态改变模型的形状，这极大地推动了不同网络的应用落地。2022年9月23日，OpenVINO<sup>TM</sup> 工具套件2022.2版推出，对2022.1进行了微调，以包括对英特尔最新 CPU 和离散 GPU 的支持，以实现更多的人工智能创新和机会。

&emsp;此处选用OpenVINO<sup>TM</sup> 2022.2 版本，对于Python本，我们可以直接使用PIP命令安装。建议使用Anaconda 创建虚拟环境安装，对于最新版，在创建好的虚拟环境下直接输入以下命令进行安装：

```
// 更新pip
python -m pip install --upgrade pip
// 安装
pip install openvino-dev[ONNX,tensorflow2]==2022.2.0
```

&emsp;安装过程中如出现下载安装包错误以及网络等原因时，可以重新运行安装命令，会继续上一次的安装。

# 2. 创建推理类 Predictor

```python
from openvino.runtime import Core
class Predictor:
    """
    OpenVINO 模型推理器
    """
    def __init__(self, model_path):
        ie_core = Core()
        model = ie_core.read_model(model=model_path)
        self.compiled_model = ie_core.compile_model(model=model, device_name="CPU")
    def get_inputs_name(self, num):
        return self.compiled_model.input(num)
    
    def get_outputs_name(self, num):
        return self.compiled_model.output(num)
    
    def predict(self, input_data):
        return self.compiled_model([input_data])
        
```

&emsp;此处由于只进行PP-YOLOE模型推理，所以只简单地封装一下Predictor类：主要包括初始化函数，负责读取本地模型并加载到指定设备中；获取输入输出名称函数以及模型预测函数。



# 3.数据处理方法

## 3.1 输入图片预处理

```python
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
```

&emsp;根据 PP-YOLOE模型输入要求，处理图片数据，主要包括图片通道转换、图片缩放、转换矩阵、数据归一化以及增加矩阵维度。按照PP-YOLOE模型输入设置，归一化方式是直接将像素点除255，将输入数据整合到0~1之间，加快模型的计算。PP-YOLOE模型ONNX格式只支持bath_size=1的推理，所以最后将数据矩阵维度直接增加一个维度即可。

## 3.2 模型输出结果处理

```python
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
    conf_results = np.transpose(conf_results,[0, 2, 1]) # 转置
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
```

&emsp;由于我们所使用的PP-YOLOE被我们裁剪过，因此模型的输出是未进行处理的结果数据，模型输出节点有两个，一个为预测框输出，一个节点为置信值输出，所以后期需要对输出结果进行处理。

&emsp;置信度结果输出形状为[1, 80, 8400]，而实际80代表的一个预测结果对应的80个类别的置信值，而8400表示有8400个预测结果；而预测框输出结果为形状为[1, 8400, 4]，对应了8400个预测结果的预测框，其中4代表预测框的左上顶点预右下顶点的横纵坐标。

&emsp;因此结果处理主要包含以下几个方面：

- 置信度结果转置处理，并提取预测结果最大的类别、预测分数和对应的预测框；
- 非极大值抑制提取预测框和类别。

## 3.3 绘制预测结果

```python
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
```

&emsp;上一步经过结果处理，最终获得预测框、分数以及类别，最后通过OpenCV将预测结果绘制到图片上，主要是一个预测框绘制和分数、类别的书写两步。



# 4. 模型推理



```python
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
```

&emsp;根据模型推理流程，最后调用模型推理类进行实现：

- 导入相关信息：主要是定义模型地址、待预测图片地址和类别文件；
- 创建模型预测器：主要初始化预测类，读取本地模型，此处可以读取ONNX模型和IR模型两种格式；
- 预处理图片：调用定义的图片处理方法，将本地图片数据转为模型推理的数据；
- 模型推理：将处理好的图片数据加载到模型中，并获取模型推理结果；
- 处理模型结果：主要是调用结果处理方法实现，如果需要可视化，可以将预测结果绘制到图片中。
