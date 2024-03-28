#include "linear_assignment.h"
#include <map>

linear_assignment *linear_assignment::instance = nullptr;
linear_assignment::linear_assignment()
= default;

linear_assignment *linear_assignment::getInstance()
{
    if(instance == nullptr) instance = new linear_assignment();
    return instance;
}

TRACHER_MATCHD
linear_assignment::matching_cascade(
        DeepsortTracker *distance_metric,
        DeepsortTracker::GATED_METRIC_FUNC distance_metric_func,
        float max_distance,
        int cascade_depth,
        std::vector<DTrack> &tracks,
        const DETECTIONS &detections,
        std::vector<int>& track_indices,
        std::vector<int> detection_indices)
{
    TRACHER_MATCHD res;
    //!!!python diff: track_indices will never be None.
    //    if(track_indices.empty() == true) {
    //        for(size_t i = 0; i < tracks.size(); i++) {
    //            track_indices.push_back(i);
    //        }
    //    }

    //!!!python diff: detection_indices will always be None.
    for(size_t i = 0; i < detections.size(); i++) {
        detection_indices.push_back(int(i));
    }

    std::vector<int> unmatched_detections;
    unmatched_detections.assign(detection_indices.begin(), detection_indices.end());
    res.matches.clear();
    std::vector<int> track_indices_l;

    std::map<int, int> matches_trackid;
    for(int level = 0; level < cascade_depth; level++) {
        if(unmatched_detections.empty()) break; //No detections left;

        track_indices_l.clear();
        for(int k:track_indices) {
            if(tracks[k].time_since_update == 1+level)
                track_indices_l.push_back(k);
        }
        if(track_indices_l.empty()) continue; //Nothing to match at this level.

        TRACHER_MATCHD tmp = min_cost_matching(
                    distance_metric, distance_metric_func,
                    max_distance, tracks, detections, track_indices_l,
                    unmatched_detections);
        unmatched_detections.assign(tmp.unmatched_detections.begin(), tmp.unmatched_detections.end());
        for(auto pa : tmp.matches) {
            res.matches.push_back(pa);
            matches_trackid.insert(pa);
        }
    }
    res.unmatched_detections.assign(unmatched_detections.begin(), unmatched_detections.end());
    for(int tid : track_indices) {
        if(matches_trackid.find(tid) == matches_trackid.end())
            res.unmatched_tracks.push_back(tid);
    }
    return res;
}

TRACHER_MATCHD
linear_assignment::min_cost_matching(
        DeepsortTracker *distance_metric,
        DeepsortTracker::GATED_METRIC_FUNC distance_metric_func,
        float max_distance,
        std::vector<DTrack> &tracks,
        const DETECTIONS &detections,
        std::vector<int> &track_indices,
        std::vector<int> &detection_indices)
{
    TRACHER_MATCHD res;
    //!!!python diff: track_indices && detection_indices will never be None.
    //    if(track_indices.empty() == true) {
    //        for(size_t i = 0; i < tracks.size(); i++) {
    //            track_indices.push_back(i);
    //        }
    //    }
    //    if(detection_indices.empty() == true) {
    //        for(size_t i = 0; i < detections.size(); i++) {
    //            detection_indices.push_back(int(i));
    //        }
    //    }
    if((detection_indices.empty()) || (track_indices.empty())) {
        res.matches.clear();
        res.unmatched_tracks.assign(track_indices.begin(), track_indices.end());
        res.unmatched_detections.assign(detection_indices.begin(), detection_indices.end());
        return res;
    }
    DYNAMICM cost_matrix = (distance_metric->*(distance_metric_func))(
                tracks, detections, track_indices, detection_indices);
    for(int i = 0; i < cost_matrix.rows(); i++) {
        for(int j = 0; j < cost_matrix.cols(); j++) {
            float tmp = cost_matrix(i,j);
            if(tmp > max_distance) cost_matrix(i,j) = max_distance + 1e-5;
        }
    }
    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> indices = DeepsortTracker::SolveHungarian(cost_matrix);
    res.matches.clear();
    res.unmatched_tracks.clear();
    res.unmatched_detections.clear();
    for(size_t col = 0; col < detection_indices.size(); col++) {
        bool flag = false;
        for(int i = 0; i < indices.rows(); i++)
            if(indices(i, 1) == col) { flag = true; break;}
        if(flag == false)res.unmatched_detections.push_back(detection_indices[col]);
    }
    for(size_t row = 0; row < track_indices.size(); row++) {
        bool flag = false;
        for(int i = 0; i < indices.rows(); i++)
            if(indices(i, 0) == row) { flag = true; break; }
        if(flag == false) res.unmatched_tracks.push_back(track_indices[row]);
    }
    for(int i = 0; i < indices.rows(); i++) {
        int row = indices(i, 0);
        int col = indices(i, 1);

        int track_idx = track_indices[row];
        int detection_idx = detection_indices[col];
        if(cost_matrix(row, col) > max_distance) {
            res.unmatched_tracks.push_back(track_idx);
            res.unmatched_detections.push_back(detection_idx);
        } else res.matches.emplace_back(track_idx, detection_idx);
    }
    return res;
}

DYNAMICM
linear_assignment::gate_cost_matrix(
        KalmanFilter* kf,
        DYNAMICM &cost_matrix,
        std::vector<DTrack> &tracks,
        const DETECTIONS &detections,
        const std::vector<int> &track_indices,
        const std::vector<int> &detection_indices,
        float gated_cost, bool only_position)
{
    int gating_dim = (only_position == true?2:4);
    double gating_threshold = KalmanFilter::chi2inv95[gating_dim];
    std::vector<DETECTBOX> measurements;
    for(int i:detection_indices) {
        const DETECTION_ROW& t = detections[i];
        measurements.push_back(t.to_xyah());
    }
    for(size_t i  = 0; i < track_indices.size(); i++) {
        DTrack& track = tracks[track_indices[i]];
        Eigen::Matrix<float, 1, -1> gating_distance = kf->gating_distance(
                    track.mean, track.covariance, measurements, only_position);
        for (int j = 0; j < gating_distance.cols(); j++) {
            if (gating_distance(0, j) > gating_threshold)  cost_matrix(i, j) = gated_cost;
        }
    }
    return cost_matrix;
}

