# OpenVINO<sup>TM</sup>部署PP-YOLOE模型

# 1. PP-YOLOE模型

&emsp;目标检测作为计算机视觉领域的顶梁柱，不仅可以独立完成车辆、商品、缺陷检测等任务，也是人脸识别、视频分析、以图搜图等复合技术的核心模块，在自动驾驶、工业视觉、安防交通等领域的商业价值有目共睹。

&emsp;PaddleDetection为基于飞桨PaddlePaddle的端到端目标检测套件，内置30+模型算法及250+预训练模型，覆盖目标检测、实例分割、跟踪、关键点检测等方向，其中包括服务器端和移动端高精度、轻量级产业级SOTA模型、冠军方案和学术前沿算法，并提供配置化的网络模块组件、十余种数据增强策略和损失函数等高阶优化支持和多种部署方案，在打通数据处理、模型开发、训练、压缩、部署全流程的基础上，提供丰富的案例及教程，加速算法产业落地应用。

&emsp; PP-YOLOE 是PaddleDetection推出的一种高精度SOTA目标检测模型，基于PP-YOLOv2的卓越的单阶段Anchor-free模型，超越了多种流行的YOLO模型。

- 尺寸多样：PP-YOLOE根据不同应用场景设计了s/m/l/x，4个尺寸的模型来支持不同算力水平的硬件，无论是哪个尺寸，精度-速度的平衡都超越当前所有同等计算量下的YOLO模型！可以通过width multiplier和depth multiplier配置。
- 性能卓越：具体来说，PP-YOLOE-l在COCO test-dev上以精度51.4%，TRT FP16推理速度149 FPS的优异数据，相较YOLOX，精度提升1.3%，加速25%；相较YOLOv5，精度提升0.7%，加速26.8%。训练速度较PP-YOLOv2提高33%，降低模型训练成本。
- 部署友好：与此同时，PP-YOLOE在结构设计上避免使用如deformable convolution或者matrix NMS之类的特殊算子，使其能轻松适配更多硬件。当前已经完备支持NVIDIA V100、T4这样的云端GPU架构以及如Jetson系列等边缘端GPU和FPGA开发板。

# 2. OpenVINO<sup>TM</sup>

&emsp;OpenVINO<sup>TM</sup>是英特尔基于自身现有的硬件平台开发的一种可以加快高性能计算机视觉和深度学习视觉应用开发速度工具套件，用于快速开发应用程序和解决方案，以解决各种任务（包括人类视觉模拟、自动语音识别、自然语言处理和推荐系统等）。                               

![image-20221012150224745](./image/image-20221012150224745.png)



&emsp;该工具套件基于最新一代的人工神经网络，包括卷积神经网络 (CNN)、递归网络和基于注意力的网络，可扩展跨英特尔® 硬件的计算机视觉和非视觉工作负载，从而最大限度地提高性能。它通过从边缘到云部署的高性能、人工智能和深度学习推理来为应用程序加速，并且允许直接异构执行。极大的提高计算机视觉、自动语音识别、自然语言处理和其他常见任务中的深度学习性能；使用使用流行的框架（如TensorFlow，PyTorch等）训练的模型；减少资源需求，并在从边缘到云的一系列英特尔®平台上高效部署；支持在Windows与Linux系统，且官方支持编程语言为Python与C++语言。

&emsp;OpenVINO<sup>TM</sup>工具套件2022.1版于2022年3月22日正式发布，与以往版本相比发生了重大革新，提供预处理API函数、ONNX前端API、AUTO 设备插件，并且支持直接读入飞桨模型，在推理中中支持动态改变模型的形状，这极大地推动了不同网络的应用落地。2022年9月23日，OpenVINO<sup>TM</sup> 工具套件2022.2版推出，对2022.1进行了微调，以包括对英特尔最新 CPU 和离散 GPU 的支持，以实现更多的人工智能创新和机会。

#  3. PP-YOLOE模型下载与转换

## 3.1. 模型下载

&emsp;首先下载PP-YOLOE官方训练模型，该模型由PaddleDetection提供，基于COCO数据集训练，可以识别80种常见物体。此处采用的是PaddleDetection release/2.5版本，PP-YOLOE+模型，具体可以参考官方文件[PP-YOLOE](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ppyoloe/README_cn.md)。

&emsp;使用命令，导出我们要使用的模型，在命令行种依次输入以下指令，导出我们所使用的模型文件：

```shell
// 打开PaddleDetection代码文件
cd ./PaddleDetection 
// 导出指定模型
python tools/export_model.py -c configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams
```

&emsp;此处导出的是PP-YOLOE+模型，l_80e格式，导出命令输出如下图所示。

![image-20221003235248259](./image/image-20221003235248259.png)

&emsp;模型导出后可以在下述文件夹中找到该模型文件：

![image-20221003235506007](./image/image-20221003235506007.png)

&emsp;利用模型查看器可以看出该模型，包含两个输入、两个输出。

![image-20221003235554676](./image/image-20221003235554676.png)

## 3.2 模型裁剪

&emsp;直接导出的模型在OpenVINO中无法直接使用，需要对模型进行裁剪，将模型后处理过程去掉，使用下面大神的提供的工具可以直接实现对Paddle模型直接裁剪：[jiangjiajun/PaddleUtils: Some tools to operate PaddlePaddle model ](https://github.com/jiangjiajun/PaddleUtils)。

&emsp;首先克隆改代码仓到本地：

```shell
git clone https://github.com/jiangjiajun/PaddleUtils.git
```

&emsp;然后打开到该代码文件中下面的一个文件夹下，并将上一步导出的模型复制到该文加夹中

![image-20221004000350336](./image/image-20221004000350336.png)

&emsp;在命令提示符中依次输入以下命令：

```shell
// 打开指定文件
cd E:\Paddle\PaddleUtils\paddle
// 模型裁剪
python prune_paddle_model.py --model_dir ppyoloe_plus_crn_l_80e_coco --model_filename model.pdmodel --params_filename model.pdiparams --output_names tmp_16 concat_14.tmp_0 --save_dir export_model
```

&emsp;指令说明：

|      标志位       |       说明       |            输入             |
| :---------------: | :--------------: | :-------------------------: |
|    --model_dir    |   模型文件路径   | ppyoloe_plus_crn_l_80e_coco |
| --model_filename  |  静态图模型文件  |        model.pdmodel        |
| --params_filename | 模型配置文件信息 |       model.pdiparams       |
|  --output_names   |    输出节点名    |   tmp_16 concat_14.tmp_0    |
|    --save_dir     |   模型保存路径   |        export_model         |

&emsp;此处主要关注输出节点名这一项输入，由于原模型输入包含后处理这一部分，在模型部署时会出错，所以模型裁剪的主要目的就是将模型后处理这一步去掉，因此将模型输出设置为后处理开始前的模型节点，此处主要存在两个节点：

&emsp;第一个节点包含模型预测的置信度输出参数，其位置如下图所示：

![image-20221004002756096](./image/image-20221004002756096.png)

&emsp;第二个节点是模型预测狂输出节点，其位置如下图所示：

![image-20221004030109206](./image/image-20221004030109206.png)



&emsp;输入上述指令后，会获得以下结果：

![image-20221004002852962](./image/image-20221004002852962.png)

&emsp;在``export_model``文件夹下，可以获得裁剪后的模型文件：

![image-20221004002911123](./image/image-20221004002911123.png)

&emsp;使用模型查看器，可以看出导出的模型，输入输出发生了改变。模型的输入仅包含**image**一项，原有的**scale_factor**输入由于在模型中使用不到，被一并削减掉。模型的输出变成我们指定的节点输出。

![image-20221004030016832](./image/image-20221004030016832.png)





## 3.3 模型转换ONNX

&emsp;由于Paddle模型未指定bath_size大小，在使用时会出现问题，因此通过将该模型转为ONNX并指定bath_size大小，此处使用``paddle2onnx``工具便可以实现。

&emsp;在命令提示符中依次输入以下指令，将上一步导出的模型转为ONNX格式：

```
cd E:\Paddle\PaddleUtils\paddle
// 模型转换
paddle2onnx --model_dir export_model --model_filename model.pdmodel --params_filename model.pdiparams --input_shape_dict "{'image':[1,3,640,640]}" --opset_version 11 --save_file ppyoloe_plus_crn_l_80e_coco.onnx
```

&emsp;此处需要指定模型的输入形状，--input_shape_dict "{'image':[1,3,640,640]}"，其他设置按照常规设置即可，模型输出如下图所示：

![image-20221004004801198](./image/image-20221004004801198.png)



![image-20221004025931422](./image/image-20221004025931422.png)

## 3.4 转为IR格式

&emsp; IR格式模型为OpenVINO<sup>TM</sup>推理工具原生支持模型，且对模型进行了进一步优化，使得推理速度大大提升，此处我们使用OpenVINO<sup>TM</sup> 自带的模型优化工具进行转换。

&emsp;首先利用命令提示窗口打开OpenVINO<sup>TM</sup>工具路径，然后输入转换命令，在该文件夹中会生成三个转换后的文件，其输出如图所示，出现三个SUCCESS表示转换成功。

```
cd .\openvino\tools
mo --input_model ppyoloe_plus_crn_l_80e_coco.onnx
```

![image-20221012152706262](./image/image-20221012152706262.png)