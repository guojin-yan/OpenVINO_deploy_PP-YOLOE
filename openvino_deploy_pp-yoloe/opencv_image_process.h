#pragma once
#include<iostream>
#include<map>
#include<string>
#include<vector>

#include "opencv2/opencv.hpp"


class ImageProcess {
public:
	// 预处理图片
	cv::Mat image_normalize(cv::Mat& sourse_mat, cv::Size& size, int type);

	// rec文本识别处理图片
	cv::Mat image_preprocess_rec(cv::Mat& sourse_mat);

	// 处理文本检测输出
	std::vector<cv::Rect> det_result_process(std::vector<float>& data, cv::Size& size);

	// 裁剪文本检测结果区域
	std::vector<cv::Mat> cut_result_roi(cv::Mat& sourse_mat, std::vector<cv::Rect>& rects, cv::Size& size);

	

private:

	// 修改矩形大小
	cv::Rect enlarge_rect(cv::Rect& rect);


};

