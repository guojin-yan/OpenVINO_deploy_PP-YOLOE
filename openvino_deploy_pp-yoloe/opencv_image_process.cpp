#include "opencv_image_process.h"

/// <summary>
/// 预处理图像数据
/// 1.转换RGB 2.缩放图片  3.图片归一化
/// </summary>
/// <param name="sourse_mat">原图片</param>
/// <param name="size">图片大小</param>
/// <param name="type">归一化方式</param>
/// <returns></returns>
cv::Mat ImageProcess::image_normalize(cv::Mat& sourse_mat, cv::Size& size, int type) {

    cv::Mat image = sourse_mat.clone();

    cv::cvtColor(image, image, cv::COLOR_BGRA2RGB); // 将图片通道由 BGR 转为 RGB
   // 对输入图片按照tensor输入要求进行缩放
    cv::resize(image, image, size, 0, 0, cv::INTER_NEAREST);
    

    if (type == 0) {
        // 图像数据归一化，减均值mean，除以方差std
        // PaddleDetection模型使用imagenet数据集的均值 Mean = [0.485, 0.456, 0.406]和方差 std = [0.229, 0.224, 0.225]
        std::vector<float> mean_values{ 0.485 * 255, 0.456 * 255, 0.406 * 255 };
        std::vector<float> std_values{ 0.229 * 255, 0.224 * 255, 0.225 * 255 };

        std::vector<cv::Mat> rgb_channels(3);

        cv::split(image, rgb_channels); // 分离图片数据通道

        for (auto i = 0; i < rgb_channels.size(); i++) {
            //分通道依此对每一个通道数据进行归一化处理
            rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / std_values[i], (0.0 - mean_values[i]) / std_values[i]);
        }

        cv::merge(rgb_channels, image); // 合并图片数据通道
        return image;
    }
    else if (type == 1) {
        std::vector<float> mean_values{ 0.5 * 255, 0.5 * 255, 0.5 * 255 };
        std::vector<float> std_values{ 0.5 * 255, 0.5 * 255, 0.5 * 255 };
        std::vector<cv::Mat> rgb_channels(3);

        cv::split(image, rgb_channels); // 分离图片数据通道

        for (auto i = 0; i < rgb_channels.size(); i++) {
            //分通道依此对每一个通道数据进行归一化处理
            rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / std_values[i], (0.0 - mean_values[i]) / std_values[i]);
        }

        cv::merge(rgb_channels, image); // 合并图片数据通道
        return image;
    }
    else if (type == 2) {
        std::vector<float> std_values{ 1.0 * 255, 1.0 * 255, 1.0 * 255 };
        std::vector<cv::Mat> rgb_channels(3);

        cv::split(image, rgb_channels); // 分离图片数据通道

        for (auto i = 0; i < rgb_channels.size(); i++) {
            //分通道依此对每一个通道数据进行归一化处理
            rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / std_values[i]);
        }

        cv::merge(rgb_channels, image); // 合并图片数据通道
        return image;
    }
}


cv::Mat ImageProcess::yoloe_result_process(cv::Mat& sourse_mat, std::vector<float>& vector_box, std::vector<float>& vector_conf) {
    cv::Mat image = sourse_mat.clone();
    cv::Mat conf_mat = cv::Mat(80, 8400, CV_32F, vector_conf.data());
    std::cout << conf_mat.cols << "   " << conf_mat.rows << std::endl;
    conf_mat = conf_mat.t();
    std::cout << conf_mat.cols << "   " << conf_mat.rows << std::endl;

    std::vector<cv::Rect> position_boxes; // 矩形框容器
    std::vector<float> confidences; // 置信值容器
    std::vector<int> classIds; // 类别容器

    for (int i = 0; i < 8400; i++) {
        // 获取最大预测结果
        cv::Mat row = conf_mat.row(i);
        cv::Point max_point;
        double score;
        cv::minMaxLoc(row, 0, &score, 0, &max_point);

        // 置信度 0～1之间
        if (score > 0.5)
        {
            // 构建预测矩形框
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
    // 非极大值抑制
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(position_boxes, confidences, 0.25, 0.35, indexes);
    // 将预测结果绘制到原图片上
    for (size_t i = 0; i < indexes.size(); i++) {
        int index = indexes[i];
        int idx = classIds[index];
        cv::rectangle(image, position_boxes[index], cv::Scalar(0, 0, 255), 1, 8);
        cv::rectangle(image, cv::Point(position_boxes[index].tl().x, position_boxes[index].tl().y - 10),
            cv::Point(position_boxes[index].br().x, position_boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
        cv::putText(image, class_names[idx] + " " + std::to_string(confidences[idx]), 
            cv::Point(position_boxes[index].tl().x, position_boxes[index].tl().y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0));
    }


    
    return image;
}

void ImageProcess::read_class_names(std::string path_name) {
    std::ifstream infile;
    infile.open(path_name.data());   //将文件流对象与文件连接起来 
    assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 

    std::string str;
    while (getline(infile, str)) {
        class_names.push_back(str);
        str.clear();

    }
    infile.close();             //关闭文件输入流 
}
