#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include "trackDataType.h"

class KalmanFilter
{
public:
    static const double chi2inv95[10];
    KalmanFilter();
    KAL_DATA initiate(const DETECTBOX_TLWH& measurement);
    void predict(KAL_MEAN& mean, KAL_COVA& covariance);
    KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance);
    KAL_DATA update(const KAL_MEAN& mean,
                    const KAL_COVA& covariance,
                    const DETECTBOX_TLWH& measurement);

    Eigen::Matrix<float, 1, -1> gating_distance(
            const KAL_MEAN& mean,
            const KAL_COVA& covariance,
            const std::vector<DETECTBOX_TLWH>& measurements,
            bool only_position = false);

private:
    Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
    Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
    float _std_weight_position;
    float _std_weight_velocity;
};

#endif // KALMANFILTER_H
