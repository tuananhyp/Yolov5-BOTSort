#ifndef NVIDIATRT_H
#define NVIDIATRT_H

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include <functional>
#include <filesystem>
#include <fstream>
using namespace nvinfer1;

#define CHECK_CUDA(call)                                                                                                            \
    {                                                                                                                               \
        cudaError_t error = call;                                                                                                   \
        if (error != cudaSuccess)                                                                                                   \
        {                                                                                                                           \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                                                                                     \
        }                                                                                                                           \
    }

typedef struct tensor_shape
{
    int batch_size;
    int channels;
    int height;
    int width;
    int size;
};

typedef struct plugin_info
{
    bool adding_plugin;
    std::vector<std::string> ts_names;
    std::string plugins_creator;
    std::string plugins_name;
    std::string plugins_layer_name;
    std::vector<nvinfer1::PluginField> plugin_field;
};

typedef struct plugins_info
{
    bool preprocessing_gpu;
    plugin_info postprocess_plugin;
};

class NvidiaTRT
{
public:
    NvidiaTRT();
    bool init(const std::string &onnx_model, const std::string &trt_model, const std::string &quantize_mode, const std::string &calib_file, int batch_size, plugins_info m_pls_info);
    ~NvidiaTRT();
    void execute(std::vector<std::vector<std::vector<float>>> inputs_data);
    std::vector<std::vector<std::vector<float>>> get_outputs_data()
    {
        return outputs_data_batch;
    }
    size_t get_output_tensor_size();
    size_t get_input_tensor_size();
    std::vector<tensor_shape> get_input_tensor_shape();
    std::vector<tensor_shape> get_output_tensor_shape();
    cv::Size get_in_size();

    cudaStream_t get_stream(){
        return stream;
    }
    
    IExecutionContext * get_context(){
        return context;
    }

    IRuntime* get_runtime(){
        return runtime;
    }

    ICudaEngine* get_engine(){
        return engine;
    }

    void* get_input_gpu_buffers(int index);
    void* get_output_gpu_buffers(int index);

private:
    virtual std::vector<std::vector<std::vector<float>>> preprocess(std::vector<cv::Mat> images)
    {
    }
    virtual void postprocess()
    {
    }
    ICudaEngine *loadEngine(const std::string &engineFile);

private:
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *context;
    std::vector<Dims> inputDims;
    std::vector<Dims> outputDims;
    std::vector<tensor_shape> input_tensors_shape;
    std::vector<tensor_shape> output_tensors_shape;
    void **buffers;
    std::vector<std::vector<std::vector<float>>> outputs_data_batch;
    cudaStream_t stream;
    int32_t nb_bindings;
    cv::Size inSize;
    std::vector<int> bindings_input;
    std::vector<int> bindings_output;
    plugins_info m_pls_info_;
};

#endif
