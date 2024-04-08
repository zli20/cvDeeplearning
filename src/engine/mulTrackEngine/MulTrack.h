#ifndef CVDL_MULTRACK_H
#define CVDL_MULTRACK_H

#include "DeepsortTracker.h"
#include "ByteTracker.h"
#include "trackDataType.h"
#include "engine/utils/Datatype.h"
#include "DTrack.h"
#include "STrack.h"

enum TrackerType {
    BYTE_TRACKER,
    DEEPSORT_TRACKER,
};

class MulObjTrack {
private:
    ByteTracker bytetracker;
    DeepsortTracker deepsorttracker;
    std::string feature_model_path;
    TrackerType tracker_type;
public:
    explicit MulObjTrack(TrackerType tracker_type_);
    explicit MulObjTrack(TrackerType tracker_type_ = DEEPSORT_TRACKER, const std::string & feature_model_path="", float max_cosine_distance = 0.2, int nn_budget = 100, float max_iou_distance = 0.7, int max_age = 30, int n_init=3);
    explicit MulObjTrack(TrackerType tracker_type_ = BYTE_TRACKER, int frame_rate = 30, int track_buffer = 30, float track_thresh = 0.5, float high_thresh = 0.6, float match_thresh = 0.8);
    void getTrack(cv::Mat &frame, std::vector<DET_RESULT>& results);
    void drawResult(cv::Mat& frame);
};

#endif //CVDL_MULTRACK_H
