#include "MulTrack.h"
#include "FeatureTensor.h"

//MulObjTrack::MulObjTrack(int tracker_type_){
//    this->tracker_type = static_cast<TrackerType>(tracker_type_);
//    switch (tracker_type_) {
//        case BYTE_TRACKER:
//            bytetracker = ByteTracker(20, 30);
//            break;
//        case DEEPSORT_TRACKER:
//            deepsorttracker = DeepsortTracker(0.2, 100);
//            break;
//        default:
//            throw std::invalid_argument("Invalid tracker type");
//    }
//}

MulObjTrack::MulObjTrack(int tracker_type_, int frame_rate, int track_buffer, float track_thresh,
                         float high_thresh, float match_thresh) {
    if(tracker_type_ != BYTE_TRACKER){
        throw std::invalid_argument("Invalid tracker type: Parameter list should match tracker");
    }
    this->tracker_type = static_cast<TrackerType>(tracker_type_);
    bytetracker = ByteTracker(frame_rate, track_buffer, track_thresh, high_thresh, match_thresh);
}

MulObjTrack::MulObjTrack(int tracker_type_, const std::string & feature_model_path, int platform, float max_cosine_distance,
                         int nn_budget, float max_iou_distance, int max_age, int n_init) {
    if(tracker_type_ != DEEPSORT_TRACKER){
        throw std::invalid_argument("Invalid tracker type: Parameter list should match tracker");
    }
    if (!FeatureTensor::getInstance()->init(feature_model_path, platform))
    {
        std::cout << "Feature init failed" << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "Feature init succeed" << std::endl;
    }
    this->tracker_type = static_cast<TrackerType>(tracker_type_);
    deepsorttracker = DeepsortTracker(max_cosine_distance, nn_budget, max_iou_distance, max_age, n_init);
}



void MulObjTrack::getTrack(cv::Mat &frame, std::vector<DET_RESULT> &results) {
    DETECTIONS detections;
    for (auto dr: results) {
        //cv::putText(frame, classes[dr.classId], cv::Point(dr.box.tl().x+10, dr.box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .8, cv::Scalar(0, 255, 0));
        if (dr.label == 2) // 0 person 1
        {
            // cv::rectangle(frame, dr.box, cv::Scalar(255, 0, 0), 2);
            DETECTION_ROW tmpRow;
            tmpRow.tlwh = DETECTBOX_TLWH(dr.box.x, dr.box.y, dr.box.width, dr.box.height);
            tmpRow.confidence = dr.confidence;
            detections.push_back(tmpRow);
        }
    }
    switch (this->tracker_type) {
        case BYTE_TRACKER:
            bytetracker.update(detections);
            break;
        case DEEPSORT_TRACKER:
            if (FeatureTensor::getInstance()->getRectsFeature(frame, detections)) {
//                std::cout << "get feature succeed!" << std::endl;
                deepsorttracker.predict();
                deepsorttracker.update(detections);
            }else{std::cout << "get feature failed!" << std::endl;}
            break;
        default:
             throw std::logic_error("Unknown tracker type");
    }

}

void MulObjTrack::drawResult(cv::Mat &frame) {
    switch (this->tracker_type) {
        case BYTE_TRACKER:
            for (auto & output_strack : bytetracker.output_stracks)
            {
                std::vector<float> tlwh = output_strack.tlwh;
//                bool vertical = tlwh[2] / tlwh[3] > 1.6;
//                if (tlwh[2] * tlwh[3] > 20 && !vertical)
                {
                    cv::Scalar s = bytetracker.get_color(output_strack.track_id);
                    cv::putText(frame, cv::format("%d", output_strack.track_id), cv::Point(tlwh[0], tlwh[1] - 5),
                                0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                    cv::rectangle(frame, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
                }
            }
            break;
        case DEEPSORT_TRACKER:
            for(DTrack& track : deepsorttracker.tracks) {
                if(!track.is_confirmed() || track.time_since_update > 1) continue;
                DETECTBOX_TLWH bbox = track.to_tlwh();
                cv::Rect rect = cv::Rect(bbox[0], bbox[1], bbox[2], bbox[3]);

                auto color_ = cv::Scalar(37 * track.track_id % 255, 17 * track.track_id % 255, 29 * track.track_id % 255);
                rectangle(frame, rect, color_, 2);
                std::string label = cv::format("%d", track.track_id);
                cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
            }
            break;
        default:
            throw std::logic_error("Unknown tracker type");
    }

}



