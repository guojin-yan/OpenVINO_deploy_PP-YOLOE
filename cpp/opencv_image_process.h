#pragma once
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"


class ImageProcess {
public:
	// 预处理图片
	cv::Mat image_normalize(cv::Mat& sourse_mat, cv::Size& size);
	// 处理推理结果
	cv::Mat yoloe_result_process(cv::Mat& sourse_mat, std::vector<float>& vector_box, std::vector<float>& vector_conf);
	// 读取lable文件
	void read_class_names(std::string path_name);
	// 设置缩放比例
	void set_scale_factor(double scale);
private:
	// 缩放比例
	double scale_factor;
	// lable容器
	std::vector<std::string> class_names;
};

