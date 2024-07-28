# ifndef FACERECOGNIZER
# define FACERECOGNIZER
#include <fstream>
#include <sstream>
#include "yolov8.h"
//#include <cuda_provider_factory.h>  ///如果使用cuda加速，需要取消注释
////#include <onnxruntime_cxx_api.h>
#include "utile.h"
#include "engine.h"
#include "utile.h"
#include <vector>
using namespace std;

class FaceEmbdding_trt
{
public:
	FaceEmbdding_trt(std::string modelpath, const YoloV8Config &config);
    std::vector<float> detect(cv::cuda::GpuMat& srcimg,  std::vector<cv::Point2f>& face_landmark_5);
    
    std::vector<float> detect(cv::Mat& srcimg,        std::vector<cv::Point2f>& face_landmark_5);                                     
    std::vector<float> postprocess(std::vector<float> &featureVector);
	
private:
	std::unique_ptr<Engine<float>> m_trtEngine_embedding = nullptr;
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat &gpuImg, const vector<Point2f> face_landmark_5);
	//void preprocess(cv::Mat img, const std::vector<cv::Point2f> face_landmark_5);
	// Postprocess the output for pose model
    

    std::vector<float> input_image;
	int input_height;
	int input_width;
    std::vector<cv::Point2f> normed_template;

    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;

};
#endif
