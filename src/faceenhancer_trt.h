# ifndef FACEENHANCE_TRT
# define FACEENHANCE_TRT
#include <fstream>
#include <sstream>
#include "opencv2/opencv.hpp"
//#include <cuda_provider_factory.h>  ///如果使用cuda加速，需要取消注释
//#include <onnxruntime_cxx_api.h>
#include"utils.h"
#include"utile.h"
#include "buffers.h"


class FaceEnhance_trt
{
public:
	FaceEnhance_trt(const std::string &onnxModelPath, const YoloV8Config &config, int method = 0);
	cv::Mat process(cv::Mat target_img, const std::vector<cv::Point2f> target_landmark_5,samplesCommon::BufferManager &buffers);
	//cv::Mat process(cv::cuda::GpuMat gpuImbBGR, const std::vector<cv::Point2f> target_landmark_5);
	//std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat &gpuImg);

	void memoryFree();
//private:
	void preprocess(cv::Mat target_img, const std::vector<cv::Point2f> face_landmark_5, cv::Mat& affine_matrix, cv::Mat& box_mask,samplesCommon::BufferManager &buffers);

	std::unique_ptr<Engine<float>> m_trtEngine_enhance = nullptr;

	Mat verifyOutput(Mat &target_img, const samplesCommon::BufferManager& buffers, Mat& affine_matrix, Mat& box_mask);

	// Used for image preprocessing
    // YoloV8 model expects values between [0.f, 1.f] so we use the following params
    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;

	std::vector<float> input_image;
	int input_height;
	int input_width;
	std::vector<cv::Point2f> normed_template;
    const float FACE_MASK_BLUR = 0.3;
	const int FACE_MASK_PADDING[4] = {0, 0, 0, 0};

	// Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Face Enhance");
	// Ort::Session *ort_session = nullptr;
	// Ort::SessionOptions sessionOptions = Ort::SessionOptions();
	// std::vector<char*> input_names;
	// std::vector<char*> output_names;
	// std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs
	// std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
	// Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
};
#endif
