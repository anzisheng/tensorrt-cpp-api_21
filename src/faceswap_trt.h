# ifndef FACESWAP_trt
# define FACESWAP_trt
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



class SwapFace_trt
{
public:
	SwapFace_trt(string model_path, const YoloV8Config &config, int method = 0);
	cv::Mat process(cv::Mat target_img, const std::vector<float> source_face_embedding, const std::vector<cv::Point2f> target_landmark_5,samplesCommon::BufferManager &buffers);
	void memoryFree();
	std::unique_ptr<Engine<float>> m_trtEngine_faceswap = nullptr;
	//!
    //! \brief Classifies digits and verify result
    //!
    //bool verifyOutput(const samplesCommon::BufferManager& buffers);
	Mat verifyOutput(Mat &target_img, const samplesCommon::BufferManager& buffers, Mat& affine_matrix, Mat& box_mask);

	~SwapFace_trt();  // 析构函数, 释放内存
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
