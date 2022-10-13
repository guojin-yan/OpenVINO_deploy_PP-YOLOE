# OpenVINO<sup>TM</sup>部署PaddlePadle-YOLOE模型

# 1. 项目介绍

&emsp;目标检测作为计算机视觉领域的顶梁柱，不仅可以独立完成车辆、商品、缺陷检测等任务，也是人脸识别、视频分析、以图搜图等复合技术的核心模块，在自动驾驶、工业视觉、安防交通等领域的商业价值有目共睹。PaddleDetection为基于飞桨PaddlePaddle的端到端目标检测套件， PP-YOLOE 是PaddleDetection推出的一种高精度SOTA目标检测模型，基于PP-YOLOv2的卓越的单阶段Anchor-free模型，超越了多种流行的YOLO模型。

&emsp;OpenVINO<sup>TM</sup>是英特尔基于自身现有的硬件平台开发的一种可以加快高性能计算机视觉和深度学习视觉应用开发速度工具套件，用于快速开发应用程序和解决方案，以解决各种任务（包括人类视觉模拟、自动语音识别、自然语言处理和推荐系统等）。OpenVINO<sup>TM</sup>工具套件2022.1版于2022年3月22日正式发布，与以往版本相比发生了重大革新，提供预处理API函数、ONNX前端API、AUTO 设备插件，并且支持直接读入飞桨模型，在推理中中支持动态改变模型的形状，这极大地推动了不同网络的应用落地。2022年9月23日，OpenVINO<sup>TM</sup> 工具套件2022.2版推出，对2022.1进行了微调，以包括对英特尔最新 CPU 和离散 GPU 的支持，以实现更多的人工智能创新和机会。

&emsp;目前对于OpenVINO<sup>TM</sup>直接部署PP-YOLOE还存在一些问题，该项目针对部署中存在的问题，对PP-YOLOE模型进行裁剪，在不顺时模型精度基础上，打通了OpenVINO<sup>TM</sup>直接部署PP-YOLOE之路。

# 2. 项目环境

-  操作系统：Windows11
- OpenVINO：2022.2
- OpenCV：4.5.5
- Visual Studio：2022
- Python：3.9.13

# 3. 项目下载安装

&emsp;项目所使用的源码均已经在Github和Gitee上开源，

```
Github:
git clone https://github.com/guojin-yan/OpenVINO_deploy_PP-YOLOE.git

Gitee:
https://gitee.com/guojin-yan/OpenVINO_deploy_PP-YOLOE.git
```

# 4. OpenVINO<sup>TM</sup>部署PP-YOLOE模型实现

- [PP-YOLOE模型下载与转换](./doc/model_download_transformation.md)

- [PP-YOLOE模型OpenVINO部署—C++](./doc/cpp_infer.md)

- [PP-YOLOE模型OpenVINO部署—Python](./doc/python_infer.md)
