// Two-view geometry; see geometry.cpp for ORB-SLAM3 attribution.
#pragma once
#include "matching.hpp"
#include <random>
#include <string>

namespace p1 {

struct GeometryResult {
    std::string model = "none";
    float score_h = 0, score_f = 0;
    // One character ('0' or '1') per appearance candidate, in its original order.
    std::string inlier_mask;
};

GeometryResult filter_geometry(const Features& before, const Features& after,
                               const std::vector<Match>& candidates, std::mt19937& rng);
} // namespace p1
