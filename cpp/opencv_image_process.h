#pragma once
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"


class ImageProcess {
public:
	// Ԥ����ͼƬ
	cv::Mat image_normalize(cv::Mat& sourse_mat, cv::Size& size);
	// ����������
	cv::Mat yoloe_result_process(cv::Mat& sourse_mat, std::vector<float>& vector_box, std::vector<float>& vector_conf);
	// ��ȡlable�ļ�
	void read_class_names(std::string path_name);
	// �������ű���
	void set_scale_factor(double scale);
private:
	// ���ű���
	double scale_factor;
	// lable����
	std::vector<std::string> class_names;
};

