
#include "SampleOnnxMNIST.h"
#include "utile.h"

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
    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

     //save plan
    // Write the engine to disk
    const auto engineName = "gfpgan_1.4.engine.NVIDIAGeForceRTX3080.fp16.1.1";
    // SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    // if (!plan)
    // {
    //     return false;
    // }
    if(!doesFileExist(engineName))
    {   SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan)
        {
            return false;
        }    
        cout << "engine name :"<< engineName << endl;
        const auto enginePath = std::filesystem::path("./") / engineName;
        cout << "engine path :"<< enginePath.string() << endl;
        std::ofstream outfile(enginePath, std::ofstream::binary);
        outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());
        spdlog::info("Success, saved engine to {}", enginePath.string());
    }
    else
    {
        cout << "loading eng file"<<endl;

    }
    const auto enginePath = std::filesystem::path("./") / engineName;
    std::ifstream file(/*trtModelPath*/enginePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        auto msg = "Error, unable to read engine file";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }


    
    // SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    // if (!plan)
    // {
    //     return false;
    // }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(m_logger));
    if (!mRuntime)
    {
        return false;
    }
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
    mRuntime->deserializeCudaEngine(buffer.data(), buffer.size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    //ASSERT(network->getNbInputs() == 2); //ansisheng ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    //ASSERT(mInputDims.nbDims == 4);

    //ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    //ASSERT(mOutputDims.nbDims == 4); //anzisheng ASSERT(mOutputDims.nbDims == 2)

    return true;


    //return true;   
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer()
{
    std::cout << "mParams.inputTensorNames.size"<<std::endl;
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);
    std::cout << "after mParams.inputTensorNames.size"<<std::endl;


    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }
    std::cout << "mParams.inputTensorNames.size"<<mParams.inputTensorNames.size()<<std::endl;
    // Read the input data into the managed buffers
   if(mParams.inputTensorNames.size() != 2){
    std::cout << "wrong net"<<std::endl;
    //return false;
   };

    if (!processInput(buffers))
    {
        return false;
    }
     // Memcpy from host input buffers to device input buffers
    std::cout << "before copyInputToDevice" <<std::endl;
    buffers.copyInputToDevice();
    std::cout << "after executeV2" <<std::endl;
    std::cout << "before executeV2" <<std::endl;
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    std::cout << "after executeV2" <<std::endl;
    if (!status)
    {
        return false;
    }

     // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

       // Verify results
    if (!verifyOutput(buffers))
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

    const int inputH = 512;//mInputDims.d[2];
    const int inputW = 512;//mInputDims.d[3];

   
    //sample::gLogInfo << "Input:" << std::endl;

    cv::Mat crop_img  = cv::imread("box_mask.jpg");
    std::cout << "crop_img: "<< crop_img.rows <<std::endl;
    vector<cv::Mat> bgrChannels(3);
    split(crop_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / (255.0*1), 0);
    }
    std::vector<float>  input_image;
    const int image_area = 512 * 512;
    input_image.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    std::cout << "00000 " << endl;
    memcpy(input_image.data(), (float *)bgrChannels[2].data, single_chn_size); ///rgb顺序
    memcpy(input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(input_image.data() + image_area * 2, (float *)bgrChannels[0].data, single_chn_size);
    std::cout << "1111 " << endl;
    float* hostDataBuffer0 = static_cast<float*>(buffers.getHostBuffer("input"));//static_cast<float*>(buffers.mManagedBuffers[0]->hostBuffer.data());
    // for (int i = 0; i < 512 * 512*3; i++)
    // {
    //     hostDataBuffer0[i] = input_image[i];

    // }
    memcpy(hostDataBuffer0, input_image.data(), 512 * 512*3);


   
    std::cout << "5555 " << endl;

    return true;    
 
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers)
{
  const int outputSize = 512;
    cout << "mParams.outputTensorNames[0]:"<< mParams.outputTensorNames[0]<<endl;
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0])); // "output"
    std::vector<float> vdata;
    vdata.resize(512*512*3);
    float* pdata = vdata.data();
    for(int i = 0; i<512*512*3; i++ )
    {
        vdata[i] = *(output+i);        
        //std::cout << vdata[i]<<std::endl;
    }
    const int out_h = 512;//outs_shape[2];
	const int out_w = 512;//outs_shape[3];
	const int channel_step = out_h * out_w;
	Mat rmat(out_h, out_w, CV_32FC1, pdata);
	Mat gmat(out_h, out_w, CV_32FC1, pdata + channel_step);
	Mat bmat(out_h, out_w, CV_32FC1, pdata + 2 * channel_step);
    std::cout << "********* " << endl;
    std::cout << rmat.rows <<"  "<< rmat.cols <<endl;
	rmat *= 255.f;
    std::cout << "&&&&& " << endl;
    gmat *= 255.f;
	bmat *= 255.f;
    std::cout << "aaaaaaaa " << endl;
    //rmat.setTo(0, rmat < 0);
	//rmat.setTo(255, rmat > 255);
	//gmat.setTo(0, gmat < 0);
    std::cout << "bbbbbbbbbb " << endl;
	//gmat.setTo(255, gmat > 255);
	//bmat.setTo(0, bmat < 0);
	//bmat.setTo(255, bmat > 255);
    std::cout << "cccccccc " << endl;
	vector<Mat> channel_mats(3);
	channel_mats[0] = bmat;
	channel_mats[1] = gmat;
	channel_mats[2] = rmat;
    Mat result;
     std::cout << "********* " << endl;
	merge(channel_mats, result);
    imwrite("result88888.jpg", result);

    std::cout << "*++++++++++ " << endl;
    // box_mask.setTo(0, box_mask < 0);
	// box_mask.setTo(1, box_mask > 1);
    // Mat dstimg = paste_back(target_img, result, box_mask, affine_matrix);
    // imwrite("result.jpg", dstimg);



    cout << "done" <<endl;
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

