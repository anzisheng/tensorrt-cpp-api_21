#include "facerecognizer_trt.h"
#include "utile.h"
#include "engine.h"
#include "yolov8.h"
using namespace cv;
using namespace std;


FaceEmbdding_trt::FaceEmbdding_trt(string onnxModelPath, const YoloV8Config &config)
{
    // Specify options for GPU inference
    Options options;
    options.optBatchSize = 1;
    options.maxBatchSize = 1;

    options.precision = config.precision;
    options.calibrationDataDirectoryPath = config.calibrationDataDirectory;

    if (options.precision == Precision::INT8) {
        if (options.calibrationDataDirectoryPath.empty()) {
            throw std::runtime_error("Error: Must supply calibration data path for INT8 calibration");
        }
    }  
    

        // Create our TensorRT inference engine
    m_trtEngine_embedding = std::make_unique<Engine<float>>(options);

    // Build the onnx model into a TensorRT engine file, cache the file to disk, and then load the TensorRT engine file into memory.
    // If the engine file already exists on disk, this function will not rebuild but only load into memory.
    // The engine file is rebuilt any time the above Options are changed.
    auto succ = m_trtEngine_embedding->buildLoadNetwork(onnxModelPath, SUB_VALS, DIV_VALS, NORMALIZE);
    if (!succ) {
        const std::string errMsg = "Error: Unable to build or load the TensorRT engine. "
                                   "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp).";
        throw std::runtime_error(errMsg);
    }

    ////在这里就直接定义了，没有像python程序里的那样normed_template = TEMPLATES.get(template) * crop_size
    this->normed_template.emplace_back(Point2f(38.29459984, 51.69630032));
    this->normed_template.emplace_back(Point2f(73.53180016, 51.50140016));
    this->normed_template.emplace_back(Point2f(56.0252,     71.73660032));
    this->normed_template.emplace_back(Point2f(41.54929968, 92.36549952));
    this->normed_template.emplace_back(Point2f(70.72989952, 92.20409968));

}

//std::vector<float> FaceEmbdding::detect(cv::Mat& srcimg,  std::vector<cv::Point2f>& face_landmark_5)
std::vector<float> FaceEmbdding_trt::detect(cv::Mat& srcimg,        std::vector<cv::Point2f>& face_landmark_5)
{
    // first crop_img, then GPU
    cout << "ect"<<endl;
    Mat crop_img;
    warp_face_by_face_landmark_5(srcimg, crop_img, face_landmark_5, this->normed_template, Size(112, 112));
    //imwrite("faceEmbedding_gpu.jpg", crop_img);
    cout << "bedding gpu"<<endl;

    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(crop_img);

    // Call detectObjects with the GPU image
    return detect(gpuImg, face_landmark_5);
}

std::vector<float> FaceEmbdding_trt::detect(cv::cuda::GpuMat &inputImageBGR,  std::vector<cv::Point2f>& face_landmark_5)
{
    cout << "wwwwwwwwww"<<endl; 
    const auto input = preprocess(inputImageBGR, face_landmark_5);
    //cout << "after preprocess"<<endl;


    std::vector<std::vector<std::vector<float>>> featureVectors;
    auto succ = m_trtEngine_embedding->runInference(input, featureVectors);
    //cout << "after runInference"<<endl;

    //
    const auto &numOutputs = m_trtEngine_embedding->getOutputDims().size();
    //cout << "befor postprocess"<<endl;
    //cout <<"numOutputs size : " << numOutputs <<endl;
    std::vector<float> ret;
    if (numOutputs == 1) {

        std::vector<float> featureVector;
        Engine<float>::transformOutput(featureVectors, featureVector);
        ret = postprocess(featureVector);
        return ret;
    }
    return ret;
}
//std::vector<float> FaceEmbdding::postprocess(std::vector<float> &featureVector);
std::vector<float> FaceEmbdding_trt::postprocess(std::vector<float> &featureVector)
{
    const auto &outputDims = m_trtEngine_embedding->getOutputDims();
    auto numChannels = outputDims[0].d[1];
    //cout << "numChannels size:"<<numChannels<<endl;
    float *pdata = featureVector.data();
    //ofstream destFile2("embedding_cpp123.txt", ios::out); 
    //cout << "show face embedding output:\n";
	// for(int i =0; i < (int)numChannels; i++)
	// {
	// 	destFile2 << pdata[i] << " " ;
    //     cout << pdata[i] << " ";
	// }
	// destFile2.close();

    //auto numAnchors = outputDims[0].d[2];
    vector<float> embedding(numChannels);
    //memcpy(embedding.data(), pdata, len_feature*sizeof(float));
    cudaMemcpy(embedding.data(), pdata, numChannels*sizeof(float), cudaMemcpyHostToHost); //import not cudaMemcpyDeviceToHost
    //cout << "cuda copy to host:"<<endl;
    // for(int i = 0 ; i < embedding.size(); i++)
    // {
    //     cout << embedding[i] << "  ";
    // }
    // cout <<endl;

    return embedding;
}

//std::vector<std::vector<cv::cuda::GpuMat>> YoloV8::preprocess(const cv::cuda::GpuMat &gpuImg) 
std::vector<std::vector<cv::cuda::GpuMat>> FaceEmbdding_trt::preprocess(const cv::cuda::GpuMat &gpuImg, const vector<Point2f> face_landmark_5)
{
    
    // Populate the input vectors
    const auto &inputDims = m_trtEngine_embedding->getInputDims();

    // Convert the image from BGR to RGB
    cv::cuda::GpuMat rgbMat;
    cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

    auto resized = rgbMat;
    // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
    if (resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2]) {
        // Only resize if not already the right size to avoid unecessary copy
        resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2]);
    }
    // Convert to format expected by our inference engine
    // The reason for the strange format is because it supports models with multiple inputs as well as batching
    // In our case though, the model only has a single input and we are using a batch size of 1.
    //cout << "hello1"<<endl;
    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    //cout << "hello2"<<endl;
    std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};
    //cout << "hello3"<<endl;
        // These params will be used in the post-processing stage
    //m_imgHeight = rgbMat.rows;
    //m_imgWidth = rgbMat.cols;
    //m_ratio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));
    //cout << "hello4"<<endl;
    return inputs;
    


}