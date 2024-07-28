#ifndef SampleOnnxMNIST_H
#define SampleOnnxMNIST_H

#include "common.h"
#include "argsParser.h"
using samplesCommon::SampleUniquePtr;
class SampleOnnxMNIST
{
public:

SampleOnnxMNIST(){}


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