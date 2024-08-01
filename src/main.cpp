//trtexec --onnx=gfpgan_1.4.onnx --saveEngine=gfpgan_1.4.engine.Orin.fp16.1.1++ --fp16

#include "buffers.h"
//#include "faceswap_fromMNist.h"
#include "cmd_line_parser.h"
#include "logger.h"
#include "engine.h"
#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include "yolov8.h"
//#include "faceswap.h"
#include "faceswap_trt.h"
//#include "face68landmarks.h"
#include "Face68Landmarks_trt.h"
#include "facerecognizer_trt.h"
#include "faceenhancer_trt.h"
//#include "faceenhancer.h"
//#include "faceenhancer_trt.h"
//#include "faceenhancer_trt2.h"
#include "faceswap_trt.h"
#include "SampleOnnxMNIST.h"
//#include "faceswap.h"
#include "engine.h"
#include "utile.h"
#include "yolov8.h"
#include <vector>
using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
   
	   
    //tensorrt part
    YoloV8Config config;
    std::string onnxModelPath;
    std::string onnxModelPathLandmark;
    std::string inputImage = "1.jpg";
    std::string outputImage = "6.jpg";
    
    
    YoloV8 yoloV8("yoloface_8n.onnx", config); //
    Face68Landmarks_trt detect_68landmarks_net_trt("2dfan4.onnx", config);
    FaceEmbdding_trt face_embedding_net_trt("arcface_w600k_r50.onnx", config);
   
  
    //SwapFace swap_face_net("inswapper_128.onnx");
    SwapFace_trt swap_face_net_trt("inswapper_128.onnx", config);
    samplesCommon::BufferManager buffers(swap_face_net_trt.m_trtEngine_faceswap->m_engine);
    
    //samplesCommon::Args args; // 接收用户传递参数的变量
    //SampleOnnxMNIST sample(initializeSampleParams(args)); // 定义一个sample实例
    //FaceEnhance enhance_face_net("gfpgan_1.4.onnx");
    FaceEnhance_trt enhance_face_net_trt("gfpgan_1.4.onnx", config);
    //FaceEnhance_trt2 enhance_face_net_trt2("gfpgan_1.4.onnx", config);
    samplesCommon::BufferManager buffers_enhance(enhance_face_net_trt.m_trtEngine_enhance->m_engine);
    cout << "gfpgan_1.4.onnx trted"<<endl;
    preciseStopwatch stopwatch;
     // Read the input image
    cv::Mat img = cv::imread(inputImage);
    cv::Mat source_img = img.clone();
    

    std::vector<Object>objects = yoloV8.detectObjects(img);
    
    // Draw the bounding boxes on the image
#ifdef SHOW
    yoloV8.drawObjectLabels(source_img, objects);
    // Save the image to disk
    const auto outputName = inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
    cv::imwrite(outputName, source_img);
    std::cout << "Saved annotated image to: " << outputName << std::endl;
#endif
    

    
    std::vector<cv::Point2f> face_landmark_5of68_trt;
    //std::cout <<"begin to detect landmark"<<std::endl;
    std::vector<cv::Point2f> face68landmarks_trt = detect_68landmarks_net_trt.detectlandmark(img, objects[0], face_landmark_5of68_trt);
    #ifdef SHOW
    std::cout << "face68landmarks_trt size: " <<face68landmarks_trt.size()<<std::endl;
    std::cout << "face_landmark_5of68_trt size: " <<face_landmark_5of68_trt.size()<<std::endl;
        for(int i =0; i < face68landmarks_trt.size(); i++)
	{
		//destFile2 << source_face_embedding[i] << " " ;
        cout << face68landmarks_trt[i] << " ";
	}

    
    for(int i =0; i < face_landmark_5of68_trt.size(); i++)
	{
		//destFile2 << source_face_embedding[i] << " " ;
        cout << face_landmark_5of68_trt[i] << " ";
	}

    #endif

    //cout << "get embedding"<<endl;
    vector<float> source_face_embedding = face_embedding_net_trt.detect(source_img, face_landmark_5of68_trt);

    // ofstream destFile2("embedding_cpp.txt", ios::out); 
    // cout << "embedding show:"<<endl;
	// for(int i =0; i < source_face_embedding.size(); i++)
	// {
	// 	destFile2 << source_face_embedding[i] << " " ;
    //     cout << source_face_embedding[i] << " ";
	// }
	// destFile2.close();

    //cv::Mat target_img = imread(target_path);
    //cout << "next to target"<<endl;

    //target_img =
    cv::Mat target_img = cv::imread(outputImage);
    cv::Mat target_img2 =target_img.clone();

    std::vector<Object>objects_target = yoloV8.detectObjects(target_img);
    //Object  obj = objects[0];

    // Draw the bounding boxes on the image
    yoloV8.drawObjectLabels(target_img2, objects_target);

    //std::cout << "Detected " << objects_target.size() << " objects" << std::endl;

    // Save the image to disk
    const auto outputName_target = outputImage.substr(0, outputImage.find_last_of('.')) + "_annotated.jpg";
    cv::imwrite(outputName_target, target_img2);
    // std::cout << "Saved annotated image to: " << outputName_target << std::endl;
    //detet_face_net.detect(target_img, boxes);    
	int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	vector<Point2f> target_landmark_5(5);    
	detect_68landmarks_net_trt.detectlandmark(target_img, objects_target[position], target_landmark_5);
	// ofstream target_5landmark("target_5.txt", ios::out);
	// for(int i = 0; i < target_landmark_5.size(); i++)
	// {
	// 	target_5landmark << target_landmark_5[i].x << "  "<<target_landmark_5[i].y << "  ";
	// }
	// target_5landmark.close();

    cout << "0000swap"<<endl;

    // target_landmark_5[0].x = 380.127;
    // target_landmark_5[0].y = 555.112;
    // target_landmark_5[1].x = 556.609;  
    // target_landmark_5[1].y = 531.036;
    // target_landmark_5[2].x =   489.365;    
    // target_landmark_5[2].y = 636.938;
    // target_landmark_5[3].x =  443.68 ;       
    // target_landmark_5[3].y = 734.912;
    // target_landmark_5[4].x =  549.813;
    // target_landmark_5[4].y = 719.047;

    //read in source_face_embedding
    // fstream source_face_emb("embedding.txt", ios::in); 
    // if(!source_face_emb.is_open())
    // {
    //     cout << "cann't open the embedding.txt"<<endl;
    // }
    // for (int i = 0; i < 512; i++)
    // {
    //     float x; source_face_emb >> x;
    //     cout << i <<" "<< x <<endl;
    //     //vdata[i] = x;
    //     source_face_embedding[i]= x;

    // }
    // source_face_emb.close();



    //cv::Mat swapimg = swap_face_net.process(target_img, source_face_embedding, target_landmark_5);
    cv::Mat swapimg = swap_face_net_trt.process(target_img, source_face_embedding, target_landmark_5, buffers);
    
    cout << "swap_face_net.process end" <<endl;
    imwrite("swapimg.jpg", swapimg);
    /////////////////////////////////////////////////
    //test
    // swapimg = imread("swapimg.jpg");

    // target_landmark_5[0].x = 382.286;
    // target_landmark_5[0].y = 554.35;
    // target_landmark_5[1].x = 558.448;  
    // target_landmark_5[1].y = 531.028;
    // target_landmark_5[2].x =  491.288;    
    // target_landmark_5[2].y = 636.619;
    // target_landmark_5[3].x =  443.534;       
    // target_landmark_5[3].y = 734.555;
    // target_landmark_5[4].x =  548.897;
    // target_landmark_5[4].y = 719.394;
///////////////////////////////
    
    //Mat resultimg = enhance_face_net.process(swapimg, target_landmark_5);
    cv::Mat resultimg = enhance_face_net_trt.process(swapimg, target_landmark_5, buffers_enhance);
    //cv::Mat resultimg = enhance_face_net_trt2.process(swapimg, target_landmark_5);
    //cout << "enhance_face_net_trt2.process end" <<endl;
    imwrite("resultimgend.jpg", resultimg);

    //if (!sample.build()) // 【主要】在build方法中构建网络，返回构建网络是否成功的状态
    // {
    //     cout<<"bad build"<<endl;
    //     return 0;////sample::gLogger.reportFail(sampleTest);
    // }
    // //if (!sample.infer()) // 【主要】读取图像并进行推理，返回推理是否成功的状态
    // {
    //      cout<<"bad build"<<endl;
    //     return 0;////sample::gLogger.reportFail(sampleTest);
    // }
	
    //preciseStopwatch stopwatch;
    auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    cout << "total time is " << totalElapsedTimeMs/1000 <<" S"<<endl;

    return 0;

/*
    CommandLineArguments arguments;

    std::string logLevelStr = getLogLevelFromEnvironment();
    spdlog::level::level_enum logLevel = toSpdlogLevel(logLevelStr);
    spdlog::set_level(logLevel);

    // Parse the command line arguments
    // if (!parseArguments(argc, argv, arguments)) {
    //     return -1;
    // }

    //std::string flag = "trt_model";
    //arguments.trtModelPath = "../models/yoloface_8n.engine.Orin.fp16.1.1";
    //const std::string inputImage = "../inputs/2.jpg";




    // Specify our GPU inference configuration options
    Options options;
    // Specify what precision to use for inference
    // FP16 is approximately twice as fast as FP32.
    options.precision = Precision::FP16;
    // If using INT8 precision, must specify path to directory containing
    // calibration data.
    options.calibrationDataDirectoryPath = "";
    // Specify the batch size to optimize for.
    options.optBatchSize = 1;
    // Specify the maximum batch size we plan on running.
    options.maxBatchSize = 1;
    // Specify the directory where you want the model engine model file saved.
    options.engineFileDir = ".";

    Engine<float> engine(options);

    // Define our preprocessing code
    // The default Engine::build method will normalize values between [0.f, 1.f]
    // Setting the normalize flag to false will leave values between [0.f, 255.f]
    // (some converted models may require this).

    // For our YoloV8 model, we need the values to be normalized between
    // [0.f, 1.f] so we use the following params
    std::array<float, 3> subVals{0.f, 0.f, 0.f};
    std::array<float, 3> divVals{1.f, 1.f, 1.f};
    bool normalize = true;
    // Note, we could have also used the default values.

    // If the model requires values to be normalized between [-1.f, 1.f], use the
    // following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;

    if (!arguments.onnxModelPath.empty()) {
        // Build the onnx model into a TensorRT engine file, and load the TensorRT
        // engine file into memory.
        bool succ = engine.buildLoadNetwork(arguments.onnxModelPath, subVals, divVals, normalize);
        if (!succ) {
            throw std::runtime_error("Unable to build or load TensorRT engine.");
        }
    } else {
        // Load the TensorRT engine file directly
        bool succ = engine.loadNetwork(arguments.trtModelPath, subVals, divVals, normalize);
        if (!succ) {
            const std::string msg = "Unable to load TensorRT engine.";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
    }

    // Read the input image
    // TODO: You will need to read the input image required for your model
    //const std::string inputImage = "../inputs/2.jpg";
    auto cpuImg = cv::imread(inputImage);
    if (cpuImg.empty()) {
        const std::string msg = "Unable to read image at path: " + inputImage;
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    // Upload the image GPU memory
    cv::cuda::GpuMat img;
    img.upload(cpuImg);

    // The model expects RGB input
    cv::cuda::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // In the following section we populate the input vectors to later pass for
    // inference
    const auto &inputDims = engine.getInputDims();
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;

    // Let's use a batch size which matches that which we set the
    // Options.optBatchSize option
    size_t batchSize = options.optBatchSize;

    // TODO:
    // For the sake of the demo, we will be feeding the same image to all the
    // inputs You should populate your inputs appropriately.
    for (const auto &inputDim : inputDims) { // For each of the model inputs...
        std::vector<cv::cuda::GpuMat> input;
        for (size_t j = 0; j < batchSize; ++j) { // For each element we want to add to the batch...
            // TODO:
            // You can choose to resize by scaling, adding padding, or a combination
            // of the two in order to maintain the aspect ratio You can use the
            // Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while
            // maintain the aspect ratio (adds padding where necessary to achieve
            // this).
            auto resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(img, inputDim.d[1], inputDim.d[2]);
            // You could also perform a resize operation without maintaining aspect
            // ratio with the use of padding by using the following instead:
            //            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2],
            //            inputDim.d[1])); // TRT dims are (height, width) whereas
            //            OpenCV is (width, height)
            input.emplace_back(std::move(resized));
        }
        inputs.emplace_back(std::move(input));
    }

    // Warm up the network before we begin the benchmark
    spdlog::info("Warming up the network...");
    std::vector<std::vector<std::vector<float>>> featureVectors;
    for (int i = 0; i < 100; ++i) {
        bool succ = engine.runInference(inputs, featureVectors);
        if (!succ) {
            const std::string msg = "Unable to run inference.";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
    }

    // Benchmark the inference time
    size_t numIterations = 1;//1000;
    spdlog::info("Running benchmarks ({} iterations)...", numIterations);
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        featureVectors.clear();
        engine.runInference(inputs, featureVectors);
    }
    auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    auto avgElapsedTimeMs = totalElapsedTimeMs / numIterations / static_cast<float>(inputs[0].size());

    spdlog::info("Benchmarking complete!");
    spdlog::info("======================");
    spdlog::info("Avg time per sample: ");
    spdlog::info("Avg time per sample: {} ms", avgElapsedTimeMs);
    spdlog::info("Batch size: {}", inputs[0].size());
    spdlog::info("Avg FPS: {} fps", static_cast<int>(1000 / avgElapsedTimeMs));
    spdlog::info("======================\n");

    // Print the feature vectors
    for (size_t batch = 0; batch < featureVectors.size(); ++batch) {
        for (size_t outputNum = 0; outputNum < featureVectors[batch].size(); ++outputNum) {
            spdlog::info("Batch {}, output {}", batch, outputNum);
            std::string output;
            int i = 0;
            for (const auto &e : featureVectors[batch][outputNum]) {
                output += std::to_string(e) + " ";
                if (++i == 10) {
                    output += "...";
                    break;
                }
            }
            spdlog::info("{}", output);
        }
    }

    // TODO: If your model requires post processing (ex. convert feature vector
    // into bounding boxes) then you would do so here.

    return 0;*/
}
