#ifndef SampleOnnxMNIST_H
#define SampleOnnxMNIST_H
#include "logger.h"
#include "common.h"
#include "argsParser.h"
#include "engine.h"
using samplesCommon::SampleUniquePtr;
#include "logger.h"
#include "parserOnnxConfig.h"

//#include "NvInfer.h"
//#include <cuda_runtime_api.h>

// #include <cstdlib>
// #include <fstream>
// #include <iostream>
// #include <sstream>
using namespace nvinfer1;
// using samplesCommon::SampleUniquePtr;

// Class to extend TensorRT logger
// class Logger2 : public nvinfer1::ILogger {
//     void log(Severity severity, const char *msg) noexcept override;
// };
class SampleOnnxMNIST
{
public:
SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
    }

    SampleOnnxMNIST(){}

     //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

     //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    Logger m_logger;
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.
    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network


};

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("build/");
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "inswapper_128.onnx";
    params.inputTensorNames.push_back("target");
    params.inputTensorNames.push_back("source");
    //cout <<"params.inputTensorNames  size: " << params.inputTensorNames.size() <<endl;
    
    params.outputTensorNames.push_back("output");
    //cout <<"params.outputTensorNames  size: " << params.outputTensorNames.size() <<endl;
    params.dlaCore = args.useDLACore;
    //params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

#endif