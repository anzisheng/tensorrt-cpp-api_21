
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
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }
    // Read the input data into the managed buffers
   if(mParams.inputTensorNames.size() != 2){
    std::cout << "wrong net"<<std::endl;
    return false;
   };

    if (!processInput(buffers))
    {
        return false;
    }
    
    return true;
}


//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputH = 128;//mInputDims.d[2];
    const int inputW = 128;//mInputDims.d[3];

    cv::Mat crop_img  = cv::imread("crop_img.jpg");
    std::cout << "crop_img: "<< crop_img.rows <<std::endl;
    vector<cv::Mat> bgrChannels(3);
    split(crop_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / (255.0*1), 0);
    }
        std::vector<float>  input_image;
    const int image_area = 128 * 128;
    input_image.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    std::cout << "00000 " << endl;
    memcpy(input_image.data(), (float *)bgrChannels[2].data, single_chn_size); ///rgb顺序
    memcpy(input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(input_image.data() + image_area * 2, (float *)bgrChannels[0].data, single_chn_size);
    std::cout << "1111 " << endl;
    float* hostDataBuffer0 = static_cast<float*>(buffers.getHostBuffer("target"));//static_cast<float*>(buffers.mManagedBuffers[0]->hostBuffer.data());
    for (int i = 0; i < 128*128*3; i++)
    {
        hostDataBuffer0[i] = input_image[i];

    }

const int Embedding_ch = 512;
    std::vector<float> embedding;
    embedding.resize(Embedding_ch);
    float *pdata  = embedding.data();
     ifstream srcFile("embedding.txt", ios::in); 
     if(!srcFile.is_open())
     {
         cout << "cann't open embedding.txt"<<endl;
     }
    std::cout << "3333 " << endl;
    for (int i = 0; i < Embedding_ch; i++)
    {       
         float x; 
         srcFile >> x;
         embedding[i] = x;
        //cout <<i <<": "<< x <<std::endl;        
    }
    float* hostDataBuffer1 = static_cast<float*>(buffers.getHostBuffer("source"));
    srcFile.close();

    std::cout << "4444 " << endl;
    for (int i = 0; i < Embedding_ch; i++)
    {
        hostDataBuffer1[i] = embedding[i];
    }
    std::cout << "5555 " << endl;

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

