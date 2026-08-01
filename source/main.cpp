#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "BoTSORT.h"
#include "DataType.h"
// #include "GlobalMotionCompensation.h"
#include "track.h"
#include "yolov5_trt.h"


int main(int argc, char **argv)
{

    cv::Mat frame;
    cv::VideoCapture cap;

    // Initialize BoTSORT tracker
    std::unique_ptr<BoTSORT> tracker;

    // Initialize BoTSORT configuration 
    std::string tracker_config_path, gmc_config_path, reid_config_path, reid_onnx_model_path;
    
    tracker_config_path = "/opt/va-1.2-Linux/config/Config_BoTSORT/tracker.ini";
    gmc_config_path = "/opt/va-1.2-Linux/config/Config_BoTSORT/gmc.ini";
    reid_config_path = "/opt/va-1.2-Linux/config/Config_BoTSORT/reid.ini";
    reid_onnx_model_path = "/opt/va-1.2-Linux/config/Config_BoTSORT/osnet_x0_25_market1501.onnx";

    tracker = std::make_unique<BoTSORT>(tracker_config_path,
                                            gmc_config_path, reid_config_path,
                                            reid_onnx_model_path);

    std::string video_path = argv[1];
    std::string output_dir = "../output";
    if(!std::filesystem::exists(output_dir))
    {
       std::filesystem::create_directories(output_dir);
    }

    cap = cv::VideoCapture(video_path);
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); 

    int width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);

    // Video output filename
    std::string output_file;
    output_file = video_path.substr(
                    video_path.find_last_of('/') + 1);
    output_file = output_file.substr(0, output_file.find_last_of('.'));    
    // Initilize write video 
    cv::VideoWriter writer(
        output_dir + "/" + output_file + ".mp4",
        cv::VideoWriter::fourcc('m','p','4','v'), // codec
        fps,
        cv::Size(width, height)
    );
    
    std::shared_ptr<YoloV5TRT> yolov5trt; 

    yolov5trt = std::make_shared<YoloV5TRT>();
    const std::string m_args = "--onnxPath=../yolov5/models/onnx_models/best_v5.onnx --trtPath=../yolov5/models/trt_models/best_v5.trt --quantizeMode=fp16 --calibFile=data-calib/vehicle-detection/calib.mdb --batchSize=1 --configPath=../yolov5/models/config/vehicle.cfg --labelPath=../yolov5/models/config/vehicle.names";
    if (!yolov5trt->init(m_args))
    {
        std::cout << " init detector failed " << std::endl;
    }

    int frame_count = 0;
    while(true)
    {
        if(!cap.read(frame) || frame.empty())
        {
            std::cout << "Read video failed, video file is invalid" << std::endl;
            break;
        }

        std::vector<cv::Mat> images = {frame};

        std::vector<std::vector<yolov5_detection>> trt_detections = yolov5trt->do_infer(images);
        std::vector<float> feature_map;
        feature_map = yolov5trt->get_features_data();

        // convert ttr_detections to std::vector<Detection>
        std::vector<Detection> detections;
        for (const auto &det: trt_detections[0])
        {
            Detection detection;               
            cv::Rect  bbox(
                std::max(0, static_cast<int>(std::floor(det.m_rect.x))),
                std::max(0, static_cast<int>(std::floor(det.m_rect.y))),
                static_cast<int>(std::floor(det.m_rect.width)),
                static_cast<int>(std::floor(det.m_rect.height))
            );
            detection.bbox_tlwh = bbox;

            bbox.width  = std::min(bbox.width,  frame.cols - bbox.x);
            bbox.height = std::min(bbox.height, frame.rows - bbox.y);

            detection.bbox_tlwh = bbox;           
            detection.confidence = det.conf;
            detection.class_id = det.class_id;
            detections.push_back(detection);
        }

        // Execute tracker
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::shared_ptr<Track>> tracks =
                tracker->track(detections, frame, feature_map);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time execute track per 1 frame: " << elapsed.count() * 1000 << " " << "ms" << std::endl;

        // Draw tracking results
        for (const std::shared_ptr<Track> &track: tracks)
        {
            std::vector<float> bbox_tlwh = track->get_tlwh();
            cv::Scalar color = cv::Scalar(0, 255, 0);
        
            cv::rectangle(frame,
                        cv::Rect(static_cast<int>(bbox_tlwh[0]),
                                static_cast<int>(bbox_tlwh[1]),
                                static_cast<int>(bbox_tlwh[2]),
                                static_cast<int>(bbox_tlwh[3])),
                        color, 2);
            cv::putText(frame, std::to_string(track->track_id),
                    cv::Point(static_cast<int>(bbox_tlwh[0]),
                              static_cast<int>(bbox_tlwh[1])),
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, color, 2);
        }
        std::cout << "Writen frame " << frame_count << std::endl;
        writer.write(frame);
        frame_count++;
    }
    std::cout << "Done" << std::endl;
    cap.release();
    writer.release();
    return 0;
}

