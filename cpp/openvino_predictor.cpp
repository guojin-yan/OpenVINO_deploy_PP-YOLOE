#include "openvino_predictor.h"

// 构造函数
Predictor::Predictor(std::string& model_path, std::string& device_name) {
    p = new CoreStruct(); // 创建推理引擎指针
    p->model_ptr = p->core.read_model(model_path); // 读取推理模型
    p->compiled_model = p->core.compile_model(p->model_ptr, device_name); // 将模型加载到设备
    p->infer_request = p->compiled_model.create_infer_request(); // 创建推理请求
}


// 获取节点张量
ov::Tensor Predictor::get_tensor(std::string node_name) {
    return p->infer_request.get_tensor(node_name);
    
}

// @brief 对网络的输入为图片数据的节点进行赋值，实现图片数据输入网络
// @param input_tensor 输入节点的tensor
// @param inpt_image 输入图片数据
void Predictor::fill_tensor_data_image(ov::Tensor& input_tensor, const cv::Mat& input_image) {
    // 获取输入节点要求的输入图片数据的大小
    ov::Shape tensor_shape = input_tensor.get_shape();
    const size_t width = tensor_shape[3]; // 要求输入图片数据的宽度
    const size_t height = tensor_shape[2]; // 要求输入图片数据的高度
    const size_t channels = tensor_shape[1]; // 要求输入图片数据的维度
    // 读取节点数据内存指针
    float* input_tensor_data = input_tensor.data<float>();

    // 将图片数据填充到网络中
    // 原有图片数据为 H、W、C 格式，输入要求的为 C、H、W 格式
    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                input_tensor_data[c * width * height + h * width + w] = input_image.at<cv::Vec<float, 3>>(h, w)[c];
            }
        }
    }
}

// @brief 对网络的输入为图片数据的节点进行赋值，实现图片数据输入网络
// @param input_tensor 输入节点的tensor
// @param inpt_image 输入图片数据
void Predictor::fill_tensor_data_image(ov::Tensor& input_tensor, const std::vector<cv::Mat> input_image) {
    // 获取输入节点要求的输入图片数据的大小
    ov::Shape tensor_shape = input_tensor.get_shape();
    const size_t width = tensor_shape[3]; // 要求输入图片数据的宽度
    const size_t height = tensor_shape[2]; // 要求输入图片数据的高度
    const size_t channels = tensor_shape[1]; // 要求输入图片数据的维度
    const size_t bath_size = tensor_shape[0]; 
    // 读取节点数据内存指针
    float* input_tensor_data = input_tensor.data<float>();

    // 将图片数据填充到网络中
    // 原有图片数据为 H、W、C 格式，输入要求的为 C、H、W 格式
    for (size_t b = 0; b < bath_size; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    input_tensor_data[b * channels * height * width + c * width * height + h * width + w] = input_image[b].at<cv::Vec<float, 3>>(h, w)[c];
                }
            }
        }
    }
    
}

// 模型推理
void Predictor::infer() {
    p->infer_request.infer();
}

/// <summary>
/// 读取模型输出
/// </summary>
/// <param name="output_node_name"></param>
/// <returns></returns>
std::vector<float> Predictor::get_output_data(std::string output_node_name) {
    // 读取指定节点的tensor
    const ov::Tensor& output_tensor = p->infer_request.get_tensor(output_node_name);
    ov::Shape shape = output_tensor.get_shape();
    //std::cout << shape << std::endl;
    std::vector<int> output_shape;
    for (int i = 0; i < shape.size(); i++) {
        output_shape.push_back(shape[i]);
    }
    // 输出数据长度
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    std::vector<float> out_data;
    out_data.resize(out_num);

    // 获取网络节点数据地址
    const float* result_ptr = output_tensor.data<const float>();
    for (int i = 0; i < out_num; i++) {
        float data = *result_ptr;
        out_data[i] = data;
        result_ptr++;
        std::cout << data << "  ";
    }
    return out_data;
}