#pragma once
#include<iostream>
#include<map>
#include<string>
#include<vector>

#include "openvino/openvino.hpp"
#include "opencv2/opencv.hpp"

#include<windows.h>

// @brief 推理核心结构体
typedef struct openvino_core {
    ov::Core core; // core对象
    std::shared_ptr<ov::Model> model_ptr; // 读取模型指针
    ov::CompiledModel compiled_model; // 模型加载到设备对象
    ov::InferRequest infer_request; // 推理请求对象
} CoreStruct;

class Predictor {

public:
    // 构造函数
    Predictor(std::string& model_path, std::string& device_name);
    // 析构函数
    ~Predictor() { delete p; }

    // 获取节点张量
    ov::Tensor get_tensor(std::string node_name);
    // 填充图片数据
    void fill_tensor_data_image(ov::Tensor& input_tensor, const cv::Mat& input_image);
    void fill_tensor_data_image(ov::Tensor& input_tensor, const std::vector<cv::Mat> input_image);

    // 模型推理
    void infer();

    // 获取模型输出
    std::vector<float> get_output_data(std::string output_node_name);


private:
    CoreStruct* p;

};