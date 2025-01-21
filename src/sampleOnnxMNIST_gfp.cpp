/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include <filesystem>
#include <spdlog/spdlog.h>

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
using namespace std;
//using namespace cv;
const std::string gSampleName = "TensorRT.sample_onnx_mnist";

// Utility method for checking if a file exists on disk
inline bool doesFileExist(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxMNIST
{
public:
    SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify 存储读取的手写数字图像的具体数字的gt

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleOnnxMNIST::build()
{
    std::string engineFileDir = ".";

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
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
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
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

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        //mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
        mRuntime->deserializeCudaEngine(buffer.data(), buffer.size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1); //ansisheng ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 4); //anzisheng ASSERT(mOutputDims.nbDims == 2)

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
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0F, 127.0F);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

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
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
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

    // // Read a random digit file
    // srand(unsigned(time(nullptr)));
    // std::vector<uint8_t> fileData(inputH * inputW);
    // mNumber = rand() % 10;
    
    // readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // Print an ascii representation
    sample::gLogInfo << "Input:" << std::endl;

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
    //std::cout << "00000 " << endl;
    memcpy(input_image.data(), (float *)bgrChannels[2].data, single_chn_size); ///rgb顺序
    memcpy(input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(input_image.data() + image_area * 2, (float *)bgrChannels[0].data, single_chn_size);
    std::cout << "1111 " << endl;
    float* hostDataBuffer0 = static_cast<float*>(buffers.getHostBuffer("input"));//static_cast<float*>(buffers.mManagedBuffers[0]->hostBuffer.data());
    for (int i = 0; i < 512 * 512*3; i++)
    {
        hostDataBuffer0[i] = input_image[i];

    }


    // for (int i = 0; i < inputH * inputW; i++)
    // {
    //     //sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");

    // }
    sample::gLogInfo << std::endl;
    
    // std::cout <<"mParams.inputTensorNames[0]-----    ："<<mParams.inputTensorNames[0]<< std::endl;
    // float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("target"));
    // //float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("Input3"));
    // for (int i = 0; i < inputH * inputW; i++)
    // {
    //     //hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    // }
    /*
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
        cout <<i <<": "<< x <<std::endl;        
    }
    float* hostDataBuffer1 = static_cast<float*>(buffers.getHostBuffer("source"));
    srcFile.close();

    std::cout << "4444 " << endl;
    for (int i = 0; i < Embedding_ch; i++)
    {
        hostDataBuffer1[i] = embedding[i];
    }*/
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
    imwrite("result.jpg", result);

    std::cout << "*++++++++++ " << endl;
    // box_mask.setTo(0, box_mask < 0);
	// box_mask.setTo(1, box_mask > 1);
    // Mat dstimg = paste_back(target_img, result, box_mask, affine_matrix);
    // imwrite("result.jpg", dstimg);



    cout << "done" <<endl;
    return true;
}

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
    params.onnxFileName = "gfpgan_1.4.onnx";
    params.inputTensorNames.push_back("input");
    //params.inputTensorNames.push_back("source");
    //cout <<"params.inputTensorNames  size: " << params.inputTensorNames.size() <<endl;
    
    params.outputTensorNames.push_back("output");
    cout <<"params.outputTensorNames  size: " << params.outputTensorNames.size() <<endl;
    params.dlaCore = args.useDLACore;
    //params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args; // 接收用户传递参数的变量
    bool argsOK = samplesCommon::parseArgs(args, argc, argv); // 将main函数的参数argc和argv解释成args，返回转换是否成功的bool值    

    if (!argsOK) // 如果转换不成功，则用日志类报错并打印帮助信息，退出程序。
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help) // 如果接收的参数是请求打印帮助信息，则打印帮助信息，退出程序。
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv); // 定义一个日志类

    sample::gLogger.reportTestStart(sampleTest); // 记录日志的开始

    SampleOnnxMNIST sample(initializeSampleParams(args)); // 定义一个sample实例

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    if (!sample.build()) // 【主要】在build方法中构建网络，返回构建网络是否成功的状态
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    // if (!sample.infer()) // 【主要】读取图像并进行推理，返回推理是否成功的状态
    // {
    //     return sample::gLogger.reportFail(sampleTest);
    // }

    return sample::gLogger.reportPass(sampleTest); // 报告结束
}
