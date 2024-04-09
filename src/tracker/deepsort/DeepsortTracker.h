#ifndef DEEPSOR_TRACKER_H
#define DEEPSOR_TRACKER_H

#include <memory>
#include <vector>

#include "KalmanFilter.h"
#include "DTrack.h"

class NearNeighborDisMetric;

class DeepsortTracker
{
public:
    explicit DeepsortTracker(float max_cosine_distance = 0.2, int nn_budget = 100, float max_iou_distance = 0.7, int max_age = 30, int n_init=3);

    void predict();
    void update(const DETECTIONS& detections);
    // typedef DYNAMICM (DeepsortTracker::* GATED_METRIC_FUNC)(
    // std::vector<DTrack>& tracks,
    // const DETECTIONS& dets,
    // const std::vector<int>& track_indices,
    // const std::vector<int>& detection_indices);

    using GATED_METRIC_FUNC = DYNAMICM (DeepsortTracker::*)(
    std::vector<DTrack>& tracks,
    const DETECTIONS& dets,
    const std::vector<int>& track_indices,
    const std::vector<int>& detection_indices);

    NearNeighborDisMetric* metric;
    float max_iou_distance;
    int max_age;
    int n_init;
    KalmanFilter * kf;
    int _next_idx;

    std::vector<DTrack> tracks;

private:    
    void _match(const DETECTIONS& detections, TRACHER_MATCHD& res);
    void _initiate_track(const DETECTION_ROW& detection);

public:
        DYNAMICM gated_matric(
            std::vector<DTrack>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
        DYNAMICM iou_cost(
                std::vector<DTrack>& tracks,
                const DETECTIONS& dets,
                const std::vector<int>& track_indices,
                const std::vector<int>& detection_indices);
        static Eigen::VectorXf iou(DETECTBOX_TLWH& bbox,
                DETECTBOXSS &candidates);

    static Eigen::Matrix<float, -1, 2, Eigen::RowMajor> SolveHungarian(const DYNAMICM &cost_matrix);
};

#endif
