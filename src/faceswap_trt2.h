# ifndef FACESWAP_trt2
# define FACESWAP_trt2
#include <fstream>
#include <sstream>
#include "opencv2/opencv.hpp"
//#include <cuda_provider_factory.h>  ///如果使用cuda加速，需要取消注释
//#include <onnxruntime_cxx_api.h>
#include"utils.h"
#include "engine.h"
#include "utils.h"
#include "utile.h"
#include "buffers.h"



class SwapFace_trt2
{
public:
	SwapFace_trt2(string model_path, const YoloV8Config &config);
	cv::Mat process(cv::Mat target_img, const std::vector<float> source_face_embedding, const std::vector<cv::Point2f> target_landmark_5,samplesCommon::BufferManager &buffers);
	void memoryFree();
	std::unique_ptr<Engine<float>> m_trtEngine_faceswap2 = nullptr;
	//!
    //! \brief Classifies digits and verify result
    //!
    //bool verifyOutput(const samplesCommon::BufferManager& buffers);
	Mat verifyOutput(Mat &target_img, const samplesCommon::BufferManager& buffers, Mat& affine_matrix, Mat& box_mask);

	~SwapFace_trt();  // 析构函数, 释放内存
	///////////////////
	//!
    //! \brief Function builds the network engine
    //!
    bool build();
	//!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

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






private:
	

	
	// YoloV8 model expects values between [0.f, 1.f] so we use the following params
    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

        float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;

	void preprocess(cv::Mat target_img, const std::vector<cv::Point2f> face_landmark_5, const std::vector<float> source_face_embedding, 
	cv::Mat& affine_matrix, cv::Mat& box_mask, samplesCommon::BufferManager &buffers);
	std::vector<float> input_image;
	std::vector<float> input_embedding;
	int input_height;
	int input_width;
	const int len_feature = 512;
    float* model_matrix;
	std::vector<cv::Point2f> normed_template;
	const float FACE_MASK_BLUR = 0.3;
	const int FACE_MASK_PADDING[4] = {0, 0, 0, 0};
	const float INSWAPPER_128_MODEL_MEAN[3] = {0.0, 0.0, 0.0};
	const float INSWAPPER_128_MODEL_STD[3] = {1.0, 1.0, 1.0};

	// Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Face Swap");
	// Ort::Session *ort_session = nullptr;
	// Ort::SessionOptions sessionOptions = Ort::SessionOptions();
	// std::vector<char*> input_names;
	// std::vector<char*> output_names;
	// std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs
	// std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
	//Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
};
#endif
