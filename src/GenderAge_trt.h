# ifndef DETECT_FACE68LANDMARKS_age_trt
# define DETECT_FACE68LANDMARKS_age_trt
#include "engine.h"
#include "utils.h"
#include "utile.h"
#include <vector>
using namespace std;
using namespace cv;
class FaceGederAge_trt
{
    public:    
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    FaceGederAge_trt(const std::string &onnxModelPath, const YoloV8Config &config, int method = 0);
    vector<float> process(const cv::Mat &image);
    vector<float> process(const cv::cuda::GpuMat &inputImageBGR);




    private:
    //void preProcess();
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat &gpuImg); 

    void postProcess();
    std::unique_ptr<Engine<float>> m_trtEngine_gen_age = nullptr;
    // Used for image preprocessing
    // YoloV8 model expects values between [0.f, 1.f] so we use the following params
    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;


};
#endif