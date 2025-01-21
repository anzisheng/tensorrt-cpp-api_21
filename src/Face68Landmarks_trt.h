# ifndef DETECT_FACE68LANDMARKS_trt
# define DETECT_FACE68LANDMARKS_trt
#include "engine.h"
#include "utils.h"
#include "utile.h"
using namespace std;
using namespace cv;
class Face68Landmarks_trt
{
public:
    //Face68Landmarks(std::string modelpath);
public:    
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    Face68Landmarks_trt(const std::string &onnxModelPath, const YoloV8Config &config, int method = 0);
    
    vector<Point2f> detectlandmark(const cv::cuda::GpuMat &inputImageBGR, Object& object,vector<Point2f> &face_landmark_5of68);    
    vector<Point2f> detectlandmark(const cv::Mat &inputImageBGR,Object& object, vector<Point2f> &face_landmark_5of68) ;
    
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat &gpuImg, Object& object);
    // Postprocess the output
    std::vector<cv::Point2f> postprocess(std::vector<float> &featureVector, vector<Point2f> &face_landmark_5of68);

    cv::Mat m_srcImg;
    cv::Mat inv_affine_matrix;


private:
    std::unique_ptr<Engine<float>> m_trtEngine_landmark = nullptr;

    // Used for image preprocessing
    // YoloV8 model expects values between [0.f, 1.f] so we use the following params
    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

        float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;

    // Filter thresholds
    // const float PROBABILITY_THRESHOLD;
    // const float NMS_THRESHOLD;
    // const int TOP_K;

    // Segmentation constants
    // const int SEG_CHANNELS;
    // const int SEG_H;
    // const int SEG_W;
    // const float SEGMENTATION_THRESHOLD;

    // Object classes as strings
//    const std::vector<std::string> CLASS_NAMES;

    // Pose estimation constant
    // const int NUM_KPS;
    // const float KPS_THRESHOLD;

    // Color list for drawing objects
    // const std::vector<std::vector<float>> COLOR_LIST = {{1, 1, 1},
    //                                                     {0.098, 0.325, 0.850},
    //                                                     {0.125, 0.694, 0.929},
    //                                                     {0.556, 0.184, 0.494},
    //                                                     {0.188, 0.674, 0.466},
    //                                                     {0.933, 0.745, 0.301},
    //                                                     {0.184, 0.078, 0.635},
    //                                                     {0.300, 0.300, 0.300},
    //                                                     {0.600, 0.600, 0.600},
    //                                                     {0.000, 0.000, 1.000},
    //                                                     {0.000, 0.500, 1.000},
    //                                                     {0.000, 0.749, 0.749},
    //                                                     {0.000, 1.000, 0.000},
    //                                                     {1.000, 0.000, 0.000},
    //                                                     {1.000, 0.000, 0.667},
    //                                                     {0.000, 0.333, 0.333},
    //                                                     {0.000, 0.667, 0.333},
    //                                                     {0.000, 1.000, 0.333},
    //                                                     {0.000, 0.333, 0.667},
    //                                                     {0.000, 0.667, 0.667},
    //                                                     {0.000, 1.000, 0.667},
    //                                                     {0.000, 0.333, 1.000},
    //                                                     {0.000, 0.667, 1.000},
    //                                                     {0.000, 1.000, 1.000},
    //                                                     {0.500, 0.333, 0.000},
    //                                                     {0.500, 0.667, 0.000},
    //                                                     {0.500, 1.000, 0.000},
    //                                                     {0.500, 0.000, 0.333},
    //                                                     {0.500, 0.333, 0.333},
    //                                                     {0.500, 0.667, 0.333},
    //                                                     {0.500, 1.000, 0.333},
    //                                                     {0.500, 0.000, 0.667},
    //                                                     {0.500, 0.333, 0.667},
    //                                                     {0.500, 0.667, 0.667},
    //                                                     {0.500, 1.000, 0.667},
    //                                                     {0.500, 0.000, 1.000},
    //                                                     {0.500, 0.333, 1.000},
    //                                                     {0.500, 0.667, 1.000},
    //                                                     {0.500, 1.000, 1.000},
    //                                                     {1.000, 0.333, 0.000},
    //                                                     {1.000, 0.667, 0.000},
    //                                                     {1.000, 1.000, 0.000},
    //                                                     {1.000, 0.000, 0.333},
    //                                                     {1.000, 0.333, 0.333},
    //                                                     {1.000, 0.667, 0.333},
    //                                                     {1.000, 1.000, 0.333},
    //                                                     {1.000, 0.000, 0.667},
    //                                                     {1.000, 0.333, 0.667},
    //                                                     {1.000, 0.667, 0.667},
    //                                                     {1.000, 1.000, 0.667},
    //                                                     {1.000, 0.000, 1.000},
    //                                                     {1.000, 0.333, 1.000},
    //                                                     {1.000, 0.667, 1.000},
    //                                                     {0.000, 0.000, 0.333},
    //                                                     {0.000, 0.000, 0.500},
    //                                                     {0.000, 0.000, 0.667},
    //                                                     {0.000, 0.000, 0.833},
    //                                                     {0.000, 0.000, 1.000},
    //                                                     {0.000, 0.167, 0.000},
    //                                                     {0.000, 0.333, 0.000},
    //                                                     {0.000, 0.500, 0.000},
    //                                                     {0.000, 0.667, 0.000},
    //                                                     {0.000, 0.833, 0.000},
    //                                                     {0.000, 1.000, 0.000},
    //                                                     {0.167, 0.000, 0.000},
    //                                                     {0.333, 0.000, 0.000},
    //                                                     {0.500, 0.000, 0.000},
    //                                                     {0.667, 0.000, 0.000},
    //                                                     {0.833, 0.000, 0.000},
    //                                                     {1.000, 0.000, 0.000},
    //                                                     {0.000, 0.000, 0.000},
    //                                                     {0.143, 0.143, 0.143},
    //                                                     {0.286, 0.286, 0.286},
    //                                                     {0.429, 0.429, 0.429},
    //                                                     {0.571, 0.571, 0.571},
    //                                                     {0.714, 0.714, 0.714},
    //                                                     {0.857, 0.857, 0.857},
    //                                                     {0.741, 0.447, 0.000},
    //                                                     {0.741, 0.717, 0.314},
    //                                                     {0.000, 0.500, 0.500}};

    // const std::vector<std::vector<unsigned int>> KPS_COLORS = {
    //     {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},   {255, 128, 0},
    //     {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0}, {51, 153, 255},
    //     {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}};

    // const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
    //                                                          {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
    //                                                          {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7}};

    // const std::vector<std::vector<unsigned int>> LIMB_COLORS = {
    //     {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {255, 51, 255}, {255, 51, 255}, {255, 51, 255},
    //     {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {0, 255, 0},    {0, 255, 0},
    //     {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0}};

};



#endif