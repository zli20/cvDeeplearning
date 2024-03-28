// #pragma once
//
// #include <opencv2/opencv.hpp>
//
// #include <librealsense2/rs.hpp>
// class Realsense
// {
// public:
// 	Realsense();
// 	~Realsense();
//
// 	void initialize();
//
// 	void get_mats(cv::Mat& depth_image, cv::Mat& color_image);
// 	float get_depth_at_point(int x, int y);
// 	void get_intrinsic();
// 	rs2::depth_frame get_depth_frame();
// private:
// 	rs2::pipeline pipe_;
// 	rs2::config cfg_;
//
// 	rs2::pipeline_profile profile_;
//
// 	std::shared_ptr<rs2::depth_frame> depth_frame_;
// 	std::shared_ptr<rs2::video_frame> color_frame_;
//
// 	float get_depth_scale(rs2::device dev);
// 	rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);
// 	cv::Mat frame_to_mat(const rs2::frame& f);
// };
//
