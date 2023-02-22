#include "opencv_image_process.h"


// @brief Ԥ����ͼ�����ݣ��������¼��������裺1.ת��RGB 2.����ͼƬ  3.ͼƬ��һ��
// @param sourse_mat ԭͼƬ
// @param size ģ�������С
// @return ���ش�����ͼƬ����
cv::Mat ImageProcess::image_normalize(cv::Mat& sourse_mat, cv::Size& size) {

    cv::Mat image = sourse_mat.clone();

    cv::cvtColor(image, image, cv::COLOR_BGRA2RGB); // ��ͼƬͨ���� BGR תΪ RGB
   // ������ͼƬ����tensor����Ҫ���������
    cv::resize(image, image, size, 0, 0, cv::INTER_NEAREST);
    

    std::vector<float> std_values{ 1.0 * 255, 1.0 * 255, 1.0 * 255 };
    std::vector<cv::Mat> rgb_channels(3);
    cv::split(image, rgb_channels); // ����ͼƬ����ͨ��

    for (auto i = 0; i < rgb_channels.size(); i++) {
        //��ͨ�����˶�ÿһ��ͨ�����ݽ��й�һ������
        rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / std_values[i]);
    }

    cv::merge(rgb_channels, image); // �ϲ�ͼƬ����ͨ��
    return image;

}

// @brief ʵ�ֽ�ģ�Ͷ�ȡ�����ݰ���ָ��Ҫ����д��������Ƶ����ͼƬ��
// @param sourse_mat ԭͼƬ
// @param vector_box Ԥ�������
// @param vector_conf ����ֵ����
// @return ����ʶ������ͼƬ
cv::Mat ImageProcess::yoloe_result_process(cv::Mat& sourse_mat, std::vector<float>& vector_box, std::vector<float>& vector_conf) {
    cv::Mat image = sourse_mat.clone();
    cv::Mat conf_mat = cv::Mat(80, 8400, CV_32F, vector_conf.data());
    conf_mat = conf_mat.t(); // ����ת�ã��������״��80��8400תΪ8400��80

    std::vector<cv::Rect> position_boxes; // ���ο�����
    std::vector<float> confidences; // ����ֵ����
    std::vector<int> classIds; // �������

    for (int i = 0; i < 8400; i++) {
        // ��ȡ���Ԥ����
        cv::Mat row = conf_mat.row(i);
        cv::Point max_point;
        double score;
        cv::minMaxLoc(row, 0, &score, 0, &max_point);

        // ���Ŷ� 0��1֮��
        if (score > 0.5)
        {
            // ����Ԥ����ο�
            float x1 = vector_box[4*(i-1)];
            float y1 = vector_box[4 * (i - 1) + 1];
            float x2 = vector_box[4 * (i - 1) + 2];
            float y2 = vector_box[4 * (i - 1) + 3];
            int x = static_cast<int>((x1) * this->scale_factor);
            int y = static_cast<int>((y1) * this->scale_factor);
            int width = static_cast<int>((x2 - x1) * this->scale_factor);
            int height = static_cast<int>((y2 - y1) * this->scale_factor);
            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;
            position_boxes.push_back(box);
            classIds.push_back(max_point.x);
            confidences.push_back(score);
        }


    }
    // �Ǽ���ֵ����
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(position_boxes, confidences, 0.25, 0.35, indexes);
    // ��Ԥ�������Ƶ�ԭͼƬ��
    for (size_t i = 0; i < indexes.size(); i++) {
        int index = indexes[i];
        int idx = classIds[index];
        cv::rectangle(image, position_boxes[index], cv::Scalar(0, 0, 255), 1, 8);
        cv::rectangle(image, cv::Point(position_boxes[index].tl().x, position_boxes[index].tl().y - 10),
            cv::Point(position_boxes[index].br().x, position_boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
        cv::putText(image, class_names[idx] + " " + std::to_string(confidences[idx]), 
            cv::Point(position_boxes[index].tl().x, position_boxes[index].tl().y -5),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255));
    }


    
    return image;
}

void ImageProcess::read_class_names(std::string path_name) {
    std::ifstream infile;
    infile.open(path_name.data());   //���ļ����������ļ��������� 
    assert(infile.is_open());   //��ʧ��,�����������Ϣ,����ֹ�������� 

    std::string str;
    while (getline(infile, str)) {
        class_names.push_back(str);
        str.clear();

    }
    infile.close();             //�ر��ļ������� 
}

void ImageProcess::set_scale_factor(double scale) {
    this->scale_factor = scale;
}
