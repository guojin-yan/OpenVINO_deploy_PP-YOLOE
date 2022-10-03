#pragma once
#include<iostream>
#include<map>
#include<string>
#include<vector>

#include "openvino/openvino.hpp"
#include "opencv2/opencv.hpp"

#include<windows.h>

// @brief ������Ľṹ��
typedef struct openvino_core {
    ov::Core core; // core����
    std::shared_ptr<ov::Model> model_ptr; // ��ȡģ��ָ��
    ov::CompiledModel compiled_model; // ģ�ͼ��ص��豸����
    ov::InferRequest infer_request; // �����������
} CoreStruct;

class Predictor {

public:
    // ���캯��
    Predictor(std::string& model_path, std::string& device_name);
    // ��������
    ~Predictor() { delete p; }

    // ��ȡ�ڵ�����
    ov::Tensor get_tensor(std::string node_name);
    // ���ͼƬ����
    void fill_tensor_data_image(ov::Tensor& input_tensor, const cv::Mat& input_image);
    void fill_tensor_data_image(ov::Tensor& input_tensor, const std::vector<cv::Mat> input_image);

    // ģ������
    void infer();

    // ��ȡģ�����
    std::vector<float> get_output_data(std::string output_node_name);


private:
    CoreStruct* p;

};