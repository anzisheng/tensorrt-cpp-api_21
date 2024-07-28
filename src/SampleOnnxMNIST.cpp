
#include "SampleOnnxMNIST.h"

bool SampleOnnxMNIST::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    //auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
     if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        //= SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    return true;   
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(3)); //sample::gLogger.getReportableSeverity())-->3
    if (!parsed)
    {
        return false;
    }

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    // if (mParams.int8)
    // {
    //     config->setFlag(BuilderFlag::kINT8);
    //     samplesCommon::setAllDynamicRanges(network.get(), 127.0F, 127.0F);
    // }

    // samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

