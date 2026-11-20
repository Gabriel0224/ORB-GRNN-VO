// Adapted from the 2D part of ORB-SLAM3 TwoViewReconstruction.cc at
// 4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4 (Normalize, ComputeH21/F21,
// CheckHomography/Fundamental and the score-ratio rule in Reconstruct).
// Copyright (C) 2014-2016 Raul Mur-Artal, Jose M.M. Montiel, Juan D. Tardos;
// Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez,
// Jose M.M. Montiel, Juan D. Tardos, University of Zaragoza.
// GPL-3.0-or-later; see LICENSE-GPL-3.0.txt.
#include "geometry.hpp"
#include <Eigen/LU>
#include <Eigen/SVD>
#include <limits>
#include <numeric>
#include <utility>

namespace p1 {
namespace {
using Eigen::Matrix3f;
using Eigen::Vector3f;
constexpr int sample_size = 8;
constexpr int iterations = 200;
constexpr float h_threshold = 5.991f;
constexpr float f_threshold = 3.841f;
constexpr float score_cap = 5.991f; // Both models, at the upstream sigma of one pixel.

struct Pair {
    Vector3f a, b; // measured pixels
    Vector3f an, bn; // normalized coordinates used only to fit hypotheses
};

// Upstream centers each image's full keypoint set, then scales each axis by
// the inverse mean absolute deviation. Scoring still uses measured pixels.
struct Normalization {
    Matrix3f transform;
    Eigen::Vector2f mean;

    Vector3f apply(const cv::Point2f& point) const {
        // Match upstream's subtract-then-scale order for points near a cutoff.
        return {(point.x - mean.x()) * transform(0,0),
                (point.y - mean.y()) * transform(1,1), 1};
    }
};

bool normalization(const Features& features, Normalization& normalized) {
    if (features.keys.empty()) return false;
    Eigen::Vector2f mean = Eigen::Vector2f::Zero();
    for (const auto& k : features.keys) mean += Eigen::Vector2f(k.pt.x, k.pt.y);
    mean /= static_cast<float>(features.keys.size());
    Eigen::Vector2f deviation = Eigen::Vector2f::Zero();
    for (const auto& k : features.keys)
        deviation += (Eigen::Vector2f(k.pt.x, k.pt.y) - mean).cwiseAbs();
    deviation /= static_cast<float>(features.keys.size());
    if (!deviation.allFinite() || deviation.minCoeff() <= 0) return false;
    normalized.mean = mean;
    Matrix3f& transform = normalized.transform;
    transform = Matrix3f::Identity();
    transform(0,0) = 1 / deviation.x();
    transform(1,1) = 1 / deviation.y();
    transform(0,2) = -mean.x() * transform(0,0);
    transform(1,2) = -mean.y() * transform(1,1);
    return transform.allFinite();
}

struct Fit {
    float score = 0;
    std::string mask;
};

// Keep separate directional score contributions as in ORB-SLAM3. A match
// enters the mask only when BOTH directions satisfy the model's threshold.
template <class Errors>
void score_hypothesis(const std::vector<Pair>& pairs, float threshold, Errors errors,
                      std::string& mask, Fit& best) {
    float score = 0;
    for (size_t i = 0; i < pairs.size(); ++i) {
        const auto error = errors(pairs[i]);
        const bool first = error.first <= threshold, second = error.second <= threshold;
        if (first) score += score_cap - error.first;
        if (second) score += score_cap - error.second;
        mask[i] = first && second ? '1' : '0';
    }
    if (score > best.score) {
        best.score = score;
        best.mask = mask;
    }
}

float transfer_error(const Matrix3f& h, const Vector3f& from, const Vector3f& observed) {
    const float w = h(2,0)*from.x() + h(2,1)*from.y() + h(2,2);
    if (w == 0) return std::numeric_limits<float>::infinity();
    const float inv_w = 1.f / w;
    const float u = (h(0,0)*from.x() + h(0,1)*from.y() + h(0,2)) * inv_w;
    const float v = (h(1,0)*from.x() + h(1,1)*from.y() + h(1,2)) * inv_w;
    const float du = observed.x()-u, dv = observed.y()-v;
    return du*du + dv*dv;
}

float epipolar_error(const Matrix3f& f, const Vector3f& from, const Vector3f& point) {
    const float a = f(0,0)*from.x() + f(0,1)*from.y() + f(0,2);
    const float b = f(1,0)*from.x() + f(1,1)*from.y() + f(1,2);
    const float c = f(2,0)*from.x() + f(2,1)*from.y() + f(2,2);
    const float denominator = a*a + b*b;
    if (!(denominator > 0)) return std::numeric_limits<float>::infinity();
    const float residual = a*point.x() + b*point.y() + c;
    return residual * residual / denominator;
}

// The right null vector is the normalized DLT solution.
bool null_matrix(const Eigen::MatrixXf& design, unsigned int options, bool require_rank_eight,
                 Matrix3f& matrix) {
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(design, options);
    if (require_rank_eight && svd.rank() < sample_size) return false;
    matrix = Eigen::Map<const Eigen::Matrix<float,3,3,Eigen::RowMajor>>(svd.matrixV().col(8).data());
    return matrix.allFinite();
}

void try_homography(const Eigen::MatrixXf& design, const Matrix3f& t1, const Matrix3f& t2_inverse,
                    const std::vector<Pair>& pairs, std::string& scratch, Fit& best) {
    Matrix3f normalized;
    if (!null_matrix(design, Eigen::ComputeFullV, true, normalized)) return;
    const Matrix3f h = t2_inverse * normalized * t1;
    if (!h.allFinite() || h.determinant() == 0) return;
    const Matrix3f inverse = h.inverse();
    if (!inverse.allFinite()) return;
    score_hypothesis(pairs, h_threshold, [&](const Pair& p) {
        return std::make_pair(transfer_error(inverse, p.b, p.a), transfer_error(h, p.a, p.b));
    }, scratch, best);
}

void try_fundamental(const Eigen::MatrixXf& design, const Matrix3f& t1, const Matrix3f& t2,
                     const std::vector<Pair>& pairs, std::string& scratch, Fit& best) {
    Matrix3f preliminary;
    // For a supported image pair the upstream eight-point solver also scores
    // samples of near-planar points. Rejecting them here changes the winner.
    if (!null_matrix(design, Eigen::ComputeFullU | Eigen::ComputeFullV, false, preliminary)) return;
    Eigen::JacobiSVD<Matrix3f> svd(preliminary, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.rank() < 2) return;
    Vector3f singular = svd.singularValues();
    singular.z() = 0;
    const Matrix3f normalized = svd.matrixU() * singular.asDiagonal() * svd.matrixV().transpose();
    const Matrix3f f = t2.transpose() * normalized * t1;
    const Matrix3f transpose = f.transpose();
    if (!f.allFinite()) return;
    score_hypothesis(pairs, f_threshold, [&](const Pair& p) {
        return std::make_pair(epipolar_error(f, p.a, p.b), epipolar_error(transpose, p.b, p.a));
    }, scratch, best);
}
} // namespace

GeometryResult filter_geometry(const Features& before, const Features& after,
                               const std::vector<Match>& candidates, std::mt19937& rng) {
    GeometryResult result;
    result.inlier_mask.assign(candidates.size(), '0');
    if (candidates.size() < sample_size) return result;
    Normalization a_norm, b_norm;
    if (!normalization(before, a_norm) || !normalization(after, b_norm)) return result;
    const Matrix3f& t1 = a_norm.transform;
    const Matrix3f& t2 = b_norm.transform;
    const Matrix3f t2_inverse = t2.inverse();
    std::vector<Pair> pairs;
    pairs.reserve(candidates.size());
    for (const auto& match : candidates) {
        const auto& a = before.keys[match.previous].pt;
        const auto& b = after.keys[match.current].pt;
        const Vector3f first(a.x, a.y, 1), second(b.x, b.y, 1);
        pairs.push_back({first, second, a_norm.apply(a), b_norm.apply(b)});
    }
    Fit homography, fundamental;
    Eigen::MatrixXf all_f(pairs.size(), 9);
    for (size_t i = 0; i < pairs.size(); ++i) {
        const Pair& p = pairs[i];
        const float x = p.an.x(), y = p.an.y(), u = p.bn.x(), v = p.bn.y();
        all_f.row(i) << u*x, u*y, u, v*x, v*y, v, x, y, 1;
    }
    // If even the complete candidate set cannot constrain F, only H is eligible.
    const bool can_fit_f = all_f.colPivHouseholderQr().rank() >= sample_size;
    std::string scratch(pairs.size(), '0');
    std::vector<size_t> indices(pairs.size());
    Eigen::MatrixXf h_design(2 * sample_size, 9), f_design(sample_size, 9);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::iota(indices.begin(), indices.end(), 0);
        for (int j = 0; j < sample_size; ++j) {
            std::uniform_int_distribution<size_t> draw(j, indices.size() - 1);
            std::swap(indices[j], indices[draw(rng)]);
            const Pair& p = pairs[indices[j]];
            const float x = p.an.x(), y = p.an.y(), u = p.bn.x(), v = p.bn.y();
            h_design.row(2*j)   << 0, 0, 0, -x, -y, -1, v*x, v*y, v;
            h_design.row(2*j+1) << x, y, 1, 0, 0, 0, -u*x, -u*y, -u;
            f_design.row(j) = all_f.row(indices[j]);
        }
        try_homography(h_design, t1, t2_inverse, pairs, scratch, homography);
        if (can_fit_f) try_fundamental(f_design, t1, t2, pairs, scratch, fundamental);
    }
    result.score_h = homography.score;
    result.score_f = fundamental.score;
    const float total = homography.score + fundamental.score;
    if (!(total > 0)) return result;
    const bool choose_h = homography.score / total > 0.50f;
    const Fit& best = choose_h ? homography : fundamental;
    if (std::count(best.mask.begin(), best.mask.end(), '1') < sample_size) return result;
    result.model = choose_h ? "H" : "F";
    result.inlier_mask = best.mask;
    return result;
}
} // namespace p1
