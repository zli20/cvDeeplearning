// #include "Realsens.h"
//
// Realsense::Realsense()
// {
// }
//
// Realsense::~Realsense()
// {
//     pipe_.stop();
// }
//
// void Realsense::initialize()
// {
//     cfg_.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
//     cfg_.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
//     float depth_scale_;
//
//     profile_ = pipe_.start(cfg_);
//     depth_scale_ = get_depth_scale(profile_.get_device());
//
//     rs2_stream align_to = find_stream_to_align(profile_.get_streams());
//     rs2::align align(align_to);
//
//     float depth_clipping_distance = 3.f;
//     rs2::spatial_filter spat_filter;
//     rs2::hole_filling_filter Hole_Filling_filter(1);//������˲���
//     rs2::decimation_filter decimationFilter;
//     spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.55f);
// }
//
// void Realsense::get_mats(cv::Mat& depth_mat, cv::Mat& color_mat)
// {
//     rs2::colorizer color_map_;
//     rs2::frameset frameset = pipe_.wait_for_frames();
//
//     depth_frame_ = std::make_shared<rs2::depth_frame>(frameset.get_depth_frame());
//     color_frame_ = std::make_shared<rs2::video_frame>(frameset.get_color_frame());
//
//     auto depth_frame = depth_frame_->apply_filter(color_map_);
//     //rs2::depth_frame depth_frame = *depth_frame_;
//     //rs2::frame filtered_depth_frame = depth_frame.apply_filter(color_map_);
//
//     color_mat = frame_to_mat(*color_frame_);
//     depth_mat = frame_to_mat(depth_frame);
// }
//
// float Realsense::get_depth_at_point(int x, int y)
// {
//     float depth_value = depth_frame_->get_distance(x, y);
//
//     return depth_value;
// }
//
// void Realsense::get_intrinsic()
// {
//     auto _stream_depth = profile_.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
//     const rs2_intrinsics depth_intrin = _stream_depth.get_intrinsics();
// }
//
// rs2::depth_frame Realsense::get_depth_frame()
// {
//     return *depth_frame_;
// }
//
// float Realsense::get_depth_scale(rs2::device dev)
// {
//     // Go over the device's sensors
//     for (rs2::sensor& sensor : dev.query_sensors())
//     {
//         // Check if the sensor if a depth sensor
//         if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
//         {
//             return dpt.get_depth_scale();
//         }
//     }
//     throw std::runtime_error("Device does not have a depth sensor");
// }
//
// rs2_stream Realsense::find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
// {
//     rs2_stream align_to = RS2_STREAM_ANY;
//     bool depth_stream_found = false;
//     bool color_stream_found = false;
//     for (rs2::stream_profile sp : streams)
//     {
//         rs2_stream profile_stream = sp.stream_type();
//         if (profile_stream != RS2_STREAM_DEPTH)
//         {
//             if (!color_stream_found)         //Prefer color
//                 align_to = profile_stream;
//
//             if (profile_stream == RS2_STREAM_COLOR)
//             {
//                 color_stream_found = true;
//             }
//         }
//         else
//         {
//             depth_stream_found = true;
//         }
//     }
//
//     if (!depth_stream_found)
//         throw std::runtime_error("No Depth stream available");
//
//     if (align_to == RS2_STREAM_ANY)
//         throw std::runtime_error("No stream found to align with Depth");
//
//     return align_to;
// }
//
// cv::Mat Realsense::frame_to_mat(const rs2::frame& f)
// {
//
//     auto vf = f.as<rs2::video_frame>();
//     const int w = vf.get_width();
//     const int h = vf.get_height();
//
//     if (f.get_profile().format() == RS2_FORMAT_BGR8)
//     {
//         return cv::Mat(cv::Size(w, h), CV_8UC3, (void*)f.get_data(), cv::Mat::AUTO_STEP);
//     }
//     else if (f.get_profile().format() == RS2_FORMAT_RGB8)
//     {
//         auto r = cv::Mat(cv::Size(w, h), CV_8UC3, (void*)f.get_data(), cv::Mat::AUTO_STEP);
//         cvtColor(r, r, cv::COLOR_RGB2BGR);
//         return r;
//     }
//     else if (f.get_profile().format() == RS2_FORMAT_Z16)
//     {
//         return cv::Mat(cv::Size(w, h), CV_16UC1, (void*)f.get_data(), cv::Mat::AUTO_STEP);
//     }
//     else if (f.get_profile().format() == RS2_FORMAT_Y8)
//     {
//         return cv::Mat(cv::Size(w, h), CV_8UC1, (void*)f.get_data(), cv::Mat::AUTO_STEP);
//     }
//
//     throw std::runtime_error("Frame format is not supported yet!");
// }