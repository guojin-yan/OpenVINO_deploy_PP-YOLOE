// openvino_deploy_pp-yoloe.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>

#include "openvino/openvino.hpp"
#include "opencv2/opencv.hpp"

#include "opencv_image_process.h"
#include "openvino_predictor.h"


void openvino_deploy_ppyoloe();

int main()
{
    std::cout << "Hello World!\n";
    openvino_deploy_ppyoloe();
}

void openvino_deploy_ppyoloe() {
    // 模型路径
    std::string model_path = "../model/ppyoloe_plus_crn_s_80e_coco.onnx";
    // 设备名称
    std::string device_name = "CPU";
    // 输入节点
    std::string input__node_name = "image";
    // 输出节点名
    std::string output_box_node_name = "tmp_16";
    std::string output_conf_node_name = "concat_14.tmp_0";

    // 测试图片
    std::string image_path = "../image/demo_1.jpg";
    // 类别文件
    std::string lable_path = "../model/lable.txt";

    // 创建推理通道
    Predictor predictor(model_path, device_name);

    ImageProcess image_pro;

    image_pro.read_class_names(lable_path);

    cv::Mat image = cv::imread(image_path);
    std::cout << "Hello World!\n";
    cv::Size input_size(640, 640);
    int length = image.rows > image.cols ? image.rows : image.cols;
    cv::Mat input_mat = cv::Mat::zeros(length, length,CV_8SC3);
    cv::Rect roi(0, 0, image.cols, image.rows);
    std::cout << "Hello World!\n";
    image.copyTo(input_mat(roi));
    std::cout << "Hello World!\n";
    image_pro.scale_factor = (double)length / 640.0 ;
    std::cout << "Hello World!\n"<< image_pro.scale_factor<<"\n";
    cv::Mat input_data = image_pro.image_normalize(image, input_size, 0);
    std::cout << "Hello World!\n";
    ov::Tensor input_tensor = predictor.get_tensor(input__node_name);

    predictor.fill_tensor_data_image(input_tensor, input_data);
    predictor.infer();
    std::cout << "Hello World!\n";
    std::vector<float> result_boxes = predictor.get_output_data(output_box_node_name);
    std::vector<float> result_conf = predictor.get_output_data(output_conf_node_name);
    std::cout << "Hello World!\n";
    cv::Mat result_image = image_pro.yoloe_result_process(image, result_boxes, result_conf);

    cv::imshow("result", result_image);
    cv::waitKey(0);





}


