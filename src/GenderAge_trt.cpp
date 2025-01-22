#include "GenderAge_trt.h"
#include "engine.h"
#include <iostream>
#include <fstream>
using namespace std;
using namespace std;
using namespace cv;

FaceGederAge_trt::FaceGederAge_trt(const std::string &onnxModelPath, const YoloV8Config &config, int method)
{
    cout << "Face68Landmarks_age_trt "<< onnxModelPath <<endl;
    
    // Specify options for GPU inference
    Options options;
    options.optBatchSize = 1;
    options.maxBatchSize = 1;

    options.precision = config.precision;
    options.calibrationDataDirectoryPath = config.calibrationDataDirectory;

    if (options.precision == Precision::INT8) {
        if (options.calibrationDataDirectoryPath.empty()) {
            throw std::runtime_error("Error: Must supply calibration data path for INT8 calibration");
        }
    }  
    
    
        // Create our TensorRT inference engine
    m_trtEngine_gen_age = std::make_unique<Engine<float>>(options);

    // Build the onnx model into a TensorRT engine file, cache the file to disk, and then load the TensorRT engine file into memory.
    // If the engine file already exists on disk, this function will not rebuild but only load into memory.
    // The engine file is rebuilt any time the above Options are changed.
    auto succ = m_trtEngine_gen_age->buildLoadNetwork(onnxModelPath, SUB_VALS, DIV_VALS, NORMALIZE);
    if (!succ) {
        const std::string errMsg = "Error: Unable to build or load the TensorRT engine. "
                                   "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp).";
        throw std::runtime_error(errMsg);
    }
}

vector<float> FaceGederAge_trt::process(const cv::Mat &imageBGR)
{
    // Upload the image to GPU memory
    std::cout << "into image cpu"<< std::endl;
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(imageBGR);
    std::cout << "into gpu"<< std::endl;
    return process(gpuImg);

}


std::vector<std::vector<cv::cuda::GpuMat>> FaceGederAge_trt::preprocess(const cv::cuda::GpuMat &gpuImg) {
    // Populate the input vectors
    const auto &inputDims = m_trtEngine_gen_age->getInputDims();
    std::cout << "into gpu preprocess"<< std::endl;

    // Convert the image from BGR to RGB
    cv::cuda::GpuMat rgbMat;
    cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);
    
    std::cout << "into gpu cvtColor"<< std::endl;
    auto resized = rgbMat;

    // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
    if (resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2]) {
        // Only resize if not already the right size to avoid unecessary copy
        resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2]);
    }

    // Convert to format expected by our inference engine
    // The reason for the strange format is because it supports models with multiple inputs as well as batching
    // In our case though, the model only has a single input and we are using a batch size of 1.
    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    
    std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};

    // These params will be used in the post-processing stage
    m_imgHeight = rgbMat.rows;
    m_imgWidth = rgbMat.cols;
    m_ratio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));

    return inputs;
}



vector<float> FaceGederAge_trt::process(const cv::cuda::GpuMat &inputImageBGR){
    //handle the input image into network input.
    const auto input = preprocess(inputImageBGR);

    //<<<>>> used to store output
    std::vector<std::vector<std::vector<float>>> featureVectors;
    //do network inference.
    auto succ = m_trtEngine_gen_age->runInference(input, featureVectors);
    if (!succ) {
        throw std::runtime_error("Error: Unable to run inference.");
    }

    //std::vector<Object> ret;
    std::vector<float> ret;
    const auto &numOutputs = m_trtEngine_gen_age->getOutputDims().size(); // new output is 1x3
    if (numOutputs == 1) {
        // Object detection or pose estimation
        // Since we have a batch size of 1 and only 1 output, we must convert the output from a 3D array to a 1D array.
        std::vector<float> featureVector;
        Engine<float>::transformOutput(featureVectors, featureVector);
        
        const auto &outputDims = m_trtEngine_gen_age->getOutputDims();
        int numChannels = outputDims[outputDims.size() - 1].d[1]; //get 3 channel
        std::cout << "output channels:"<<numChannels <<std::endl;
        // TODO: Need to improve this to make it more generic (don't use magic number).
        // For now it works with Ultralytics pretrained models.
        //if (numChannels == 56) {
            // Pose estimation
            //ret = postprocessPose(featureVector);
        //} 
        //else 
        {
            // Object detection
            //ret = postprocessDetect(featureVector);
            std::cout << "get gender and age" <<std::endl;
            for(int i = 0; i< featureVector.size(); i++)
            {
                std::cout << featureVector[i]<<std::endl;
            }
        }
    }


}
