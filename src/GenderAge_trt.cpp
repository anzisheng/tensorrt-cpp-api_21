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
    //std::vector<nvinfer1::Dims3> m_inputDims;
    //[[nodiscard]] const std::vector<nvinfer1::Dims3> &getInputDims() const override { return m_inputDims; };
    const auto &inputDims = m_trtEngine_gen_age->getInputDims();
    //inputDims[0] reprsent for channel
    //inputDims[0].d[1] represent for Height
    //inputDims[0].d[2] represent for Width
    std::cout << "into gpu preprocess, dims is :"<<inputDims[0]<<","<<inputDims[0].d[1] <<","<<inputDims[0].d[2] << std::endl;

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
    //std::cout << "input channels: "<< input.size() <<std::endl;
    
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

        std::vector<float> featureVector;
        Engine<float>::transformOutput(featureVectors, featureVector);
        ret = postprocess(featureVector);
        return ret;
    }
    return ret;


}

std::vector<float> FaceGederAge_trt::postprocess(std::vector<float> &featureVector)
{
    const auto &outputDims = m_trtEngine_gen_age->getOutputDims();
    auto numChannels = outputDims[0].d[1];
    cout << "age numChannels size:"<<numChannels<<endl;
    float *pdata = featureVector.data();
    //ofstream destFile2("embedding_cpp123.txt", ios::out); 
    //cout << "show face embedding output:\n";
	// for(int i =0; i < (int)numChannels; i++)
	// {
	// 	destFile2 << pdata[i] << " " ;
    //     cout << pdata[i] << " ";
	// }
	// destFile2.close();

    //auto numAnchors = outputDims[0].d[2];
    vector<float> embedding(numChannels);
    //memcpy(embedding.data(), pdata, len_feature*sizeof(float));
    cudaMemcpy(embedding.data(), pdata, numChannels*sizeof(float), cudaMemcpyHostToHost); //import not cudaMemcpyDeviceToHost
    //cout << "cuda copy to host:"<<endl;
    for(int i = 0 ; i < embedding.size(); i++)
    {
       cout << embedding[i] << "  ";
    }
    cout <<endl;

    return embedding;
}