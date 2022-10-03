#pragma once
#include<iostream>
#include<map>
#include<string>
#include<vector>

#include "opencv2/opencv.hpp"


class ImageProcess {
public:
	// Ԥ����ͼƬ
	cv::Mat image_normalize(cv::Mat& sourse_mat, cv::Size& size, int type);

	// rec�ı�ʶ����ͼƬ
	cv::Mat image_preprocess_rec(cv::Mat& sourse_mat);

	// �����ı�������
	std::vector<cv::Rect> det_result_process(std::vector<float>& data, cv::Size& size);

	// �ü��ı����������
	std::vector<cv::Mat> cut_result_roi(cv::Mat& sourse_mat, std::vector<cv::Rect>& rects, cv::Size& size);

	

private:

	// �޸ľ��δ�С
	cv::Rect enlarge_rect(cv::Rect& rect);


};

