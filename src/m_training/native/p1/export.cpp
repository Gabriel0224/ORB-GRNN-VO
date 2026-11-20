// SPDX-License-Identifier: GPL-3.0-or-later
#include "ORBextractor.h"
#include "geometry.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using Clock=std::chrono::steady_clock;
static void check(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
static void version() {
    std::cout<<"{\"name\":\"P1\",\"backend\":\"p1_open_loop\",\"schema_version\":2,"
        "\"geometry_filter\":\"orbslam3_fh_ransac\","
        "\"upstream_commit\":\"4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4\","
        "\"opencv\":\""<<CV_VERSION<<"\",\"adapter_sha256\":\""<<P1_ADAPTER_SHA
        <<"\",\"matcher_sha256\":\""<<P1_MATCHER_SHA<<"\",\"build_sha256\":\""<<P1_BUILD_SHA
        <<"\",\"geometry_sha256\":\""<<P1_GEOMETRY_SHA<<"\",\"geometry_header_sha256\":\""<<P1_GEOMETRY_HEADER_SHA
        <<"\",\"extractor_sha256\":\""<<P1_EXTRACTOR_SHA<<"\",\"header_sha256\":\""<<P1_HEADER_SHA
        <<"\",\"pose_estimation\":false}\n";
}
int main(int argc,char** argv) {
    try {
        if (argc==2 && std::string(argv[1])=="--version") { version(); return 0; }
        if (argc!=5) throw std::runtime_error("Usage: p1_export SETTINGS IMAGE_LIST OUTPUT_JSONL SEED");
        cv::FileStorage settings(argv[1],cv::FileStorage::READ);
        check(settings.isOpened(),"Cannot read P1 settings");
        int width=int(settings["width"]), height=int(settings["height"]);
        int count=int(settings["nfeatures"]), levels=int(settings["nlevels"]);
        float scale=float(settings["scale_factor"]);
        int high=int(settings["fast_threshold"]), low=int(settings["min_fast_threshold"]);
        std::ifstream images(argv[2]); check(bool(images),"Cannot read image list");
        std::vector<std::string> filenames; std::string filename;
        while (std::getline(images,filename)) {
            if (!filename.empty() && filename.back()=='\r') filename.pop_back();
            if (!filename.empty()) filenames.push_back(filename);
        }
        int last=static_cast<int>(filenames.size())-1;
        p1::Parameters parameters;
        parameters.radius=float(settings["search_radius"]);
        parameters.ratio=float(settings["ratio_threshold"]);
        parameters.octave_tolerance=int(settings["octave_tolerance"]);
        parameters.max_hamming=int(settings["max_hamming"]);
        check(width>0 && height>0 && count>0 && levels>0 && scale>1 && high>0 && low>0 && low<=high && last>=1,
              "Invalid extraction settings");
        check(parameters.radius>0 && parameters.ratio>0 && parameters.ratio<1 && parameters.octave_tolerance>=0 &&
              parameters.max_hamming>=0 && parameters.max_hamming<=256,"Invalid matching settings");
        const int seed=std::stoi(argv[4]);
        cv::setNumThreads(1); cv::setRNGSeed(seed);
        std::mt19937 geometry_rng(seed);
        ORB_SLAM3::ORBextractor extractor(count,scale,levels,high,low);
        std::ofstream output(argv[3]); check(bool(output),"Cannot write P1 observations");
        output<<std::setprecision(9);
        p1::Features previous;
        for (int index=0; index<=last; ++index) {
            auto image=cv::imread(filenames[index],cv::IMREAD_GRAYSCALE);
            check(!image.empty(),"Missing or unreadable image");
            check(image.cols==width && image.rows==height,"All sequence images must share the calibrated size");
            auto start=Clock::now(); p1::Features current;
            std::vector<int> overlap{0,0};
            extractor(image,cv::noArray(),current.keys,current.descriptors,overlap);
            std::vector<p1::Match> matches;
            if (index) matches=p1::match(previous,current,width,height,parameters);
            const auto geometry = p1::filter_geometry(previous, current, matches, geometry_rng);
            double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            output<<"{\"input_frame\":"<<index<<",\"feature_count\":"<<current.keys.size()
                  <<",\"processing_ms\":"<<ms<<",\"keypoints\":[";
            for (size_t j=0; j<current.keys.size(); ++j) {
                if (j) output<<','; const auto& k=current.keys[j];
                output<<'['<<k.pt.x<<','<<k.pt.y<<','<<k.octave<<','<<k.angle<<']';
            }
            output<<"],\"geometry\":{\"model\":\""<<geometry.model
                  <<"\",\"score_h\":"<<geometry.score_h<<",\"score_f\":"<<geometry.score_f
                  <<",\"inlier_mask\":\""<<geometry.inlier_mask<<"\"},\"candidate_matches\":[";
            for (size_t j=0; j<matches.size(); ++j) {
                if (j) output<<','; const auto& m=matches[j];
                const auto& a=previous.keys[m.previous].pt; const auto& b=current.keys[m.current].pt;
                output<<'['<<m.previous<<','<<m.current<<','<<m.distance<<','<<a.x<<','<<a.y<<','<<b.x<<','<<b.y<<']';
            }
            output<<"]}\n";
            if (index%100==0 || index==last) {
                output.flush(); std::cout<<"P1 frame="<<index<<" features="<<current.keys.size()
                    <<" candidates="<<matches.size()<<" model="<<geometry.model<<" inliers="
                    <<std::count(geometry.inlier_mask.begin(),geometry.inlier_mask.end(),'1')<<'\n'<<std::flush;
            }
            previous=std::move(current);
        }
        output.flush(); check(bool(output),"Failed writing P1 observations");
        return 0;
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
