#include "openvino_predictor.h"

// ���캯��
Predictor::Predictor(std::string& model_path, std::string& device_name) {
    p = new CoreStruct(); // ������������ָ��
    p->model_ptr = p->core.read_model(model_path); // ��ȡ����ģ��
    p->compiled_model = p->core.compile_model(p->model_ptr, device_name); // ��ģ�ͼ��ص��豸
    p->infer_request = p->compiled_model.create_infer_request(); // ������������
}


// ��ȡ�ڵ�����
ov::Tensor Predictor::get_tensor(std::string node_name) {
    return p->infer_request.get_tensor(node_name);
    
}

// @brief �����������ΪͼƬ���ݵĽڵ���и�ֵ��ʵ��ͼƬ������������
// @param input_tensor ����ڵ��tensor
// @param inpt_image ����ͼƬ����
void Predictor::fill_tensor_data_image(ov::Tensor& input_tensor, const cv::Mat& input_image) {
    // ��ȡ����ڵ�Ҫ�������ͼƬ���ݵĴ�С
    ov::Shape tensor_shape = input_tensor.get_shape();
    const size_t width = tensor_shape[3]; // Ҫ������ͼƬ���ݵĿ��
    const size_t height = tensor_shape[2]; // Ҫ������ͼƬ���ݵĸ߶�
    const size_t channels = tensor_shape[1]; // Ҫ������ͼƬ���ݵ�ά��
    // ��ȡ�ڵ������ڴ�ָ��
    float* input_tensor_data = input_tensor.data<float>();

    // ��ͼƬ������䵽������
    // ԭ��ͼƬ����Ϊ H��W��C ��ʽ������Ҫ���Ϊ C��H��W ��ʽ
    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                input_tensor_data[c * width * height + h * width + w] = input_image.at<cv::Vec<float, 3>>(h, w)[c];
            }
        }
    }
}

// @brief �����������ΪͼƬ���ݵĽڵ���и�ֵ��ʵ��ͼƬ������������
// @param input_tensor ����ڵ��tensor
// @param inpt_image ����ͼƬ����
void Predictor::fill_tensor_data_image(ov::Tensor& input_tensor, const std::vector<cv::Mat> input_image) {
    // ��ȡ����ڵ�Ҫ�������ͼƬ���ݵĴ�С
    ov::Shape tensor_shape = input_tensor.get_shape();
    const size_t width = tensor_shape[3]; // Ҫ������ͼƬ���ݵĿ��
    const size_t height = tensor_shape[2]; // Ҫ������ͼƬ���ݵĸ߶�
    const size_t channels = tensor_shape[1]; // Ҫ������ͼƬ���ݵ�ά��
    const size_t bath_size = tensor_shape[0]; 
    // ��ȡ�ڵ������ڴ�ָ��
    float* input_tensor_data = input_tensor.data<float>();

    // ��ͼƬ������䵽������
    // ԭ��ͼƬ����Ϊ H��W��C ��ʽ������Ҫ���Ϊ C��H��W ��ʽ
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

// ģ������
void Predictor::infer() {
    p->infer_request.infer();
}

/// <summary>
/// ��ȡģ�����
/// </summary>
/// <param name="output_node_name"></param>
/// <returns></returns>
std::vector<float> Predictor::get_output_data(std::string output_node_name) {
    // ��ȡָ���ڵ��tensor
    const ov::Tensor& output_tensor = p->infer_request.get_tensor(output_node_name);
    ov::Shape shape = output_tensor.get_shape();
    //std::cout << shape << std::endl;
    std::vector<int> output_shape;
    for (int i = 0; i < shape.size(); i++) {
        output_shape.push_back(shape[i]);
    }
    // ������ݳ���
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    std::vector<float> out_data;
    out_data.resize(out_num);

    // ��ȡ����ڵ����ݵ�ַ
    const float* result_ptr = output_tensor.data<const float>();
    for (int i = 0; i < out_num; i++) {
        float data = *result_ptr;
        out_data[i] = data;
        result_ptr++;
        std::cout << data << "  ";
    }
    return out_data;
}