#ifndef YOLOV5TRT_H
#define YOLOV5TRT_H

#include "nvidia_trt.h"
#include <unordered_map>

struct PreprocessResult {
    int in_h, in_w;
    float resize_ratio;
    int tx1, tx2, ty1, ty2;
};

struct yolov5_detection{
    cv::Rect2f m_rect;
    float conf;
    int class_id;
};

class YoloV5TRT : public NvidiaTRT{
    public:
        YoloV5TRT(bool using_plugin=false);
        ~YoloV5TRT();
        bool init(const std::string& args =""/*int argc, char** argv*/);
        std::vector<std::vector<yolov5_detection>> do_infer(std::vector<cv::Mat> images);
        inline const std::unordered_map<int, std::string>& get_label_map() const {
                return labelMap;
        }
        inline std::string get_label(int index) {
            if(labelMap.find(index) != labelMap.end()){
                return labelMap[index];
            }
            return "";
        }	
	int get_batch_size(){
            return m_batch_size;
        }
	std::vector<float> get_features_data(){
		return features_map;
	}
    private:
        std::vector<std::vector<std::vector<float>>> preprocess(std::vector<cv::Mat> images);
        void postprocess();

        std::vector<std::vector<yolov5_detection>> get_result(){
            return detections;
        }

        std::vector<float> preprocess_image(cv::Mat& img_input, PreprocessResult& presult);
        std::vector<int> nms(const std::vector<std::vector<float>>& boxes,const std::vector<float>& scores,float iou_threshold = 0.4f);
        void  recover_bboxes(std::vector<std::vector<float>>& bboxes,
                                PreprocessResult presult);

    private:
        float m_conf_thresh = 0.45;
        float m_nms_thresh = 0.25;
        int m_number_class;
        int m_batch_size;
	int m_curr_batch_size;
        bool m_using_plugin = false;
        std::vector<PreprocessResult> presults;
        std::vector<std::vector<yolov5_detection>> detections;
        uint8_t* img_buffer_host = nullptr;
        uint8_t* img_buffer_device = nullptr;
	std::vector<float> features_map;
        std::unordered_map<int,std::string> labelMap;
	    std::vector<std::string> label_names;

};






#endif
