#include "opencv_image_process.h"

/// <summary>
/// Ԥ����ͼ������
/// 1.ת��RGB 2.����ͼƬ  3.ͼƬ��һ��
/// </summary>
/// <param name="sourse_mat">ԭͼƬ</param>
/// <param name="size">ͼƬ��С</param>
/// <param name="type">��һ����ʽ</param>
/// <returns></returns>
cv::Mat ImageProcess::image_normalize(cv::Mat& sourse_mat, cv::Size& size, int type) {

    cv::Mat image = sourse_mat.clone();

    cv::cvtColor(image, image, cv::COLOR_BGRA2RGB); // ��ͼƬͨ���� BGR תΪ RGB
   // ������ͼƬ����tensor����Ҫ���������
    cv::resize(image, image, size, 0, 0, cv::INTER_NEAREST);
    

    if (type == 0) {
        // ͼ�����ݹ�һ��������ֵmean�����Է���std
        // PaddleDetectionģ��ʹ��imagenet���ݼ��ľ�ֵ Mean = [0.485, 0.456, 0.406]�ͷ��� std = [0.229, 0.224, 0.225]
        std::vector<float> mean_values{ 0.485 * 255, 0.456 * 255, 0.406 * 255 };
        std::vector<float> std_values{ 0.229 * 255, 0.224 * 255, 0.225 * 255 };

        std::vector<cv::Mat> rgb_channels(3);

        cv::split(image, rgb_channels); // ����ͼƬ����ͨ��

        for (auto i = 0; i < rgb_channels.size(); i++) {
            //��ͨ�����˶�ÿһ��ͨ�����ݽ��й�һ������
            rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / std_values[i], (0.0 - mean_values[i]) / std_values[i]);
        }

        cv::merge(rgb_channels, image); // �ϲ�ͼƬ����ͨ��
        return image;
    }
    else if (type == 1) {
        std::vector<float> mean_values{ 0.5 * 255, 0.5 * 255, 0.5 * 255 };
        std::vector<float> std_values{ 0.5 * 255, 0.5 * 255, 0.5 * 255 };
        std::vector<cv::Mat> rgb_channels(3);

        cv::split(image, rgb_channels); // ����ͼƬ����ͨ��

        for (auto i = 0; i < rgb_channels.size(); i++) {
            //��ͨ�����˶�ÿһ��ͨ�����ݽ��й�һ������
            rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / mean_values[i], (0.0 - mean_values[i]) / std_values[i]);
        }

        cv::merge(rgb_channels, image); // �ϲ�ͼƬ����ͨ��
        return image;
    }
}

// rec�ı�ʶ����ͼƬ
cv::Mat ImageProcess::image_preprocess_rec(cv::Mat& sourse_mat) {
    cv::Size image_size = sourse_mat.size();
    int img_W = image_size.width;
    int img_H = 32;
    double scale_size = double(img_W) / double(img_H);

    int max_W = int(scale_size * 32);
    if (scale_size * img_H > max_W) {
        img_W = max_W;
    }
    else {
        img_W = int(scale_size * img_H);
    }

    cv::cvtColor(sourse_mat, sourse_mat, cv::COLOR_BGRA2RGB); // ��ͼƬͨ���� BGR תΪ RGB
    // ������ͼƬ����tensor����Ҫ���������
    cv::resize(sourse_mat, sourse_mat, cv::Size(img_W,img_H), 0, 0, cv::INTER_NEAREST);
    std::vector<float> mean_values{ 0.5 * 255, 0.5 * 255, 0.5 * 255 };
    std::vector<float> std_values{ 0.5 * 255, 0.5 * 255, 0.5 * 255 };
    std::vector<cv::Mat> rgb_channels(3);

    cv::split(sourse_mat, rgb_channels); // ����ͼƬ����ͨ��

    for (auto i = 0; i < rgb_channels.size(); i++) {
        //��ͨ�����˶�ÿһ��ͨ�����ݽ��й�һ������
        rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / mean_values[i], (0.0 - mean_values[i]) / std_values[i]);
    }

    cv::merge(rgb_channels, sourse_mat); // �ϲ�ͼƬ����ͨ��
    return sourse_mat;
}

/// <summary>
/// ������������ʶ����
/// </summary>
/// <param name="data"></param>
/// <param name="size"></param>
/// <returns></returns>
std::vector<cv::Rect> ImageProcess::det_result_process(std::vector<float>& data, cv::Size& size) {
    int dat_size = data.size();
    std::vector<float> pred(dat_size);
    std::vector<unsigned char> cbuf(dat_size);

    for (int i = 0; i < dat_size; i++) {
        pred[i] = float(data[i]);
        cbuf[i] = (unsigned char)((data[i]) * 255);
    }

    cv::Mat cbuf_map(size.width,size.height, CV_8UC1, (unsigned char*)cbuf.data());
    cv::Mat pred_map(size.width, size.height, CV_32F, (float*)pred.data());

    const double threshold = 0.3 * 255;
    const double maxvalue = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);

    cv::Mat diff = bit_map.clone();

    //��ֵ�˲���ʴȥ�����
    cv::medianBlur(bit_map, diff, 3);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1), cv::Point(-1, -1));
    cv::erode(diff, diff, element, cv::Point(-1, -1), 1, cv::BORDER_DEFAULT, cv::Scalar());
    //cv::imshow("diff1", diff);
    //cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1));
    //cv::dilate(diff, diff, element2, cv::Point(-1, -1), 1,cv::BORDER_DEFAULT, cv::Scalar());
    //cv::imshow("diff12", diff);

    std::vector< std::vector< cv::Point> > contours;
    cv::findContours(diff, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> rects;
    for (int i = 0; i < contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        rect = enlarge_rect(rect);
        rects.push_back(rect);
    }
    return rects;
}

/// <summary>
/// ��չ���ο�Χ
/// ֱ�ӿ��߳����ľ��η�ΧС�����ڵ�һ�������ֱ�Ե
/// </summary>
/// <param name="rect"></param>
/// <returns></returns>
cv::Rect ImageProcess::enlarge_rect(cv::Rect& rect)
{
    //// ��������
    //if (rect.width > rect.height) {
    //    cv::Point point(rect.tl().x + rect.width / 2, rect.tl().y + rect.height / 2);
    //    int width;
    //    if (rect.width < 50) {
    //        width = (int)((double)rect.width * 1.5);
    //    }
    //    else {
    //        width = (int)((double)rect.width * 1.2);
    //    }
    //    
    //    int height = (int)((double)rect.height * 2.5);

    //    cv::Rect new_rect(point.x - width / 2, point.y - height / 2, width, height);
    //    return new_rect;

    //}
    //// ��������
    //else {
    //    cv::Point point(rect.tl().x + rect.width / 2, rect.tl().y + rect.height / 2);
    //    int width = (int)((double)rect.width * 2.5);
    //    int height = (int)((double)rect.height * 1.1);

    //    cv::Rect new_rect(point.x - width / 2, point.y - height / 2, width, height);
    //    return new_rect;
    //}
    
    cv::Point point(rect.tl().x + rect.width / 2, rect.tl().y + rect.height / 2);
    int width = 0;
    int height = 0;
    // �жϾ������������
    if (rect.width > rect.height)
    {
        if (rect.width < 80)
        {
            width = (int)((double)rect.width * 1.6);
        }
        else
        {
            width = (int)((double)rect.width * 1.15);
        }
        height = (int)((double)rect.height * 3);
    }
    else
    {
        if (rect.height < 80)
        {
            height = (int)((double)rect.height * 1.5);
        }
        else
        {
            height = (int)((double)rect.height * 1.1);
        }
        width = (int)((double)rect.width * 2.5);

    }
    // �жϾ��ο��Ƿ񳬱߽�
    if (point.x - width / 2 < 0)
    {
        width = width + (point.x - width / 2) * 2;
    }
    if (point.x + width / 2 > 640)
    {
        width = width + (640 - point.x - width / 2) * 2;
    }
    if (point.y - height / 2 < 0)
    {
        height = height + (point.y - height / 2) * 2;
    }
    if (point.y + height / 2 > 640)
    {
        height = height + (640 - point.y - height / 2) * 2;
    }

    cv::Rect rect_temp(point.x - width / 2 + 1, point.y - height / 2 + 1, width - 2, height - 2);

    return rect_temp;
}

/// <summary>
/// �ü��ı����������
/// </summary>
/// <param name="sourse_mat"></param>
/// <param name="rects"></param>
/// <param name="size"></param>
/// <returns></returns>
std::vector<cv::Mat> ImageProcess::cut_result_roi(cv::Mat& sourse_mat, std::vector<cv::Rect>& rects, cv::Size& size) {
    std::vector<cv::Mat> mats;
    cv::Mat image = sourse_mat.clone();
    cv::resize(image, image, size, 0, 0, cv::INTER_NEAREST);
    
    for (int i = 0; i < rects.size(); i++) {
        cv::Mat roi = image(rects[i]);
        mats.push_back(roi);
    }
    return mats;
}

