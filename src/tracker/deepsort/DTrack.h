#ifndef DEEP_TRACK_H
#define DEEP_TRACK_H

#include "dataType.h"
#include "KalmanFilter.h"

class DTrack
{
    enum TrackState {Tentative = 1, Confirmed, Deleted};

public:
    DTrack(KAL_MEAN& mean, KAL_COVA& covariance, int track_id,
          int n_init, int max_age, const FEATURE& feature);
    void predit(KalmanFilter *kf);
    void update(KalmanFilter * kf, const DETECTION_ROW &detection);
    void mark_missed();
    bool is_confirmed() const;
    bool is_deleted() const;
    bool is_tentative() const;
    DETECTBOX to_tlwh();
    int time_since_update;
    int track_id;
    FEATURESS features;
    KAL_MEAN mean;
    KAL_COVA covariance;

    int hits;
    int age;
    int _n_init;
    int _max_age;
    TrackState state;
private:
    void featuresAppendOne(const FEATURE& f);
};

#endif // DEEP_TRACK_H
