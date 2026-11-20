// P1 two-dimensional adapter derived from ORB-SLAM3 ORBmatcher.cc:
// SearchForInitialization, DescriptorDistance and ComputeThreeMaxima.
// Copyright (C) 2014-2016 Raul Mur-Artal, Jose M.M. Montiel, Juan D. Tardos;
// Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez,
// Jose M.M. Montiel, Juan D. Tardos, University of Zaragoza.
// GPL-3.0-or-later; see LICENSE-GPL-3.0.txt.
// Adaptations: plain 2D features, all octave levels (+/- tolerance), search
// centered at the previous observed pixel, and histogram of final assignments.
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace p1 {
struct Features { std::vector<cv::KeyPoint> keys; cv::Mat descriptors; };
struct Parameters {
    float radius = 100.0f, ratio = 0.9f;
    int octave_tolerance = 1, max_hamming = 50;
};
struct Match { int previous, current, distance; };

inline int descriptor_distance(const cv::Mat& a, const cv::Mat& b) {
    const auto* pa = a.ptr<std::uint32_t>();
    const auto* pb = b.ptr<std::uint32_t>();
    int result = 0;
    for (int i=0; i<8; ++i) {
        std::uint32_t v = pa[i] ^ pb[i];
        v -= (v >> 1) & 0x55555555u;
        v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
        result += (((v + (v >> 4)) & 0x0f0f0f0fu) * 0x01010101u) >> 24;
    }
    return result;
}

inline int orientation_bin(float first, float second) {
    float rotation = first-second;
    if (rotation < 0) rotation += 360.0f;
    // Preserve the pinned upstream numeric rule, not a new 360/30-bin rule.
    int bin = static_cast<int>(std::round(rotation * (1.0f / 30)));
    return bin == 30 ? 0 : bin;
}

inline std::array<int,3> three_maxima(const std::array<std::vector<int>,30>& histogram) {
    std::array<int,3> index{{-1,-1,-1}}, size{{0,0,0}};
    for (int i=0; i<30; ++i) {
        int n = static_cast<int>(histogram[i].size());
        for (int k=0; k<3; ++k) if (n > size[k]) {
            for (int j=2; j>k; --j) { size[j]=size[j-1]; index[j]=index[j-1]; }
            size[k]=n; index[k]=i; break;
        }
    }
    if (size[1] < 0.1f*size[0]) index[1]=index[2]=-1;
    else if (size[2] < 0.1f*size[0]) index[2]=-1;
    return index;
}

inline std::vector<Match> match(const Features& before, const Features& after,
                              int width, int height, const Parameters& parameters) {
    constexpr int cell=32;
    int columns=(width+cell-1)/cell, rows=(height+cell-1)/cell;
    std::vector<std::vector<int>> grid(columns*rows);
    for (int j=0; j<static_cast<int>(after.keys.size()); ++j) {
        const auto& xy=after.keys[j].pt;
        if (xy.x>=0 && xy.y>=0 && xy.x<width && xy.y<height)
            grid[int(xy.y)/cell*columns+int(xy.x)/cell].push_back(j);
    }
    std::vector<int> forward(before.keys.size(),-1), reverse(after.keys.size(),-1);
    std::vector<int> distances(after.keys.size(),INT_MAX);
    for (int i=0; i<static_cast<int>(before.keys.size()); ++i) {
        const auto& key=before.keys[i];
        const float radius=parameters.radius;
        std::vector<int> candidates;
        int x0=std::max(0,int(std::floor((key.pt.x-radius)/cell)));
        int y0=std::max(0,int(std::floor((key.pt.y-radius)/cell)));
        int x1=std::min(columns-1,int(std::floor((key.pt.x+radius)/cell)));
        int y1=std::min(rows-1,int(std::floor((key.pt.y+radius)/cell)));
        for (int y=y0; y<=y1; ++y) for (int x=x0; x<=x1; ++x)
            for (int j:grid[y*columns+x]) {
                const auto& other=after.keys[j];
                if (std::abs(other.octave-key.octave)<=parameters.octave_tolerance &&
                    std::abs(other.pt.x-key.pt.x)<radius && std::abs(other.pt.y-key.pt.y)<radius)
                    candidates.push_back(j);
            }
        std::sort(candidates.begin(),candidates.end());
        int best=INT_MAX, second=INT_MAX, selected=-1;
        for (int j:candidates) {
            int distance=descriptor_distance(before.descriptors.row(i),after.descriptors.row(j));
            if (distances[j]<=distance) continue;
            if (distance<best) { second=best; best=distance; selected=j; }
            else if (distance<second) second=distance;
        }
        if (selected>=0 && best<=parameters.max_hamming && best<parameters.ratio*second) {
            if (reverse[selected]>=0) forward[reverse[selected]]=-1;
            forward[i]=selected; reverse[selected]=i; distances[selected]=best;
        }
    }
    std::array<std::vector<int>,30> histogram;
    for (int i=0; i<static_cast<int>(forward.size()); ++i) if (forward[i]>=0)
        histogram.at(orientation_bin(before.keys[i].angle,after.keys[forward[i]].angle)).push_back(i);
    auto retained=three_maxima(histogram);
    for (int bin=0; bin<30; ++bin)
        if (std::find(retained.begin(),retained.end(),bin)==retained.end())
            for (int i:histogram[bin]) forward[i]=-1;
    std::vector<Match> result;
    for (int i=0; i<static_cast<int>(forward.size()); ++i) if (forward[i]>=0)
        result.push_back({i,forward[i],distances[forward[i]]});
    std::sort(result.begin(),result.end(),[](const Match& a,const Match& b) {
        if (a.distance!=b.distance) return a.distance<b.distance;
        if (a.previous!=b.previous) return a.previous<b.previous;
        return a.current<b.current;
    });
    return result;
}
} // namespace p1
