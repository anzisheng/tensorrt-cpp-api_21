
#include "SampleOnnxMNIST.h"

bool SampleOnnxMNIST::build()
{
    //auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));

    return true;   
}

