#include "cmd_line_parser.h"
#include "logger.h"
#include "engine.h"
#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include "yolov8.h"
#include "Face68Landmarks_trt.h"
#include "facerecognizer_trt.h"
#include "faceswap_trt.h"
#include "SampleOnnxMNIST.h"



int main(int argc, char *argv[]) {
    CommandLineArguments arguments;

    // std::string logLevelStr = getLogLevelFromEnvironment();
    // spdlog::level::level_enum logLevel = toSpdlogLevel(logLevelStr);
    // spdlog::set_level(logLevel);

    // Parse the command line arguments
    // if (!parseArguments(argc, argv, arguments)) {
    //     return -1;
    // }

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
    YoloV8Config config;
    std::string inputImage0 = "1.jpg";
    std::string outputImage = "6.jpg";
    //cv::Mat target_img = cv::imread(outputImage);
    
        YoloV8 yoloV8("yoloface_8n.onnx", config);
        std::cout <<"yoloface_8n.onnx has trted"<<std::endl;
        cv::Mat img0 = cv::imread( "1.jpg");
        cv::Mat source_img = img0.clone();

        std::vector<Object>objects = yoloV8.detectObjects(img0);
        std::cout <<"has detected objects"<<std::endl;

        // Draw the bounding boxes on the image
        yoloV8.drawObjectLabels(img0, objects);

        std::cout << "Detected " << objects.size() << " objects" << std::endl;

        //Save the image to disk
        const auto outputName = inputImage0.substr(0, inputImage0.find_last_of('.')) + "_annotated.jpg";
        cv::imwrite(outputName, img0);
        std::cout << "Saved annotated image to: " << outputName << std::endl;
    
    
        Face68Landmarks_trt detect_68landmarks_net_trt("2dfan4.onnx", config);
    
    
        FaceEmbdding_trt face_embedding_net_trt("arcface_w600k_r50.onnx", config);
        
    

    cv::Mat target_img = cv::imread(outputImage);
    std::vector<Object>objects_target = yoloV8.detectObjects(target_img);
    yoloV8.drawObjectLabels(target_img, objects_target);
    int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	vector<Point2f> target_landmark_5(5);    
    std::vector<cv::Point2f> face_landmark_5of68_trt;
      cout << "uuuuuuuuuuuu"<<endl; 
	detect_68landmarks_net_trt.detectlandmark(target_img, objects_target[position], target_landmark_5);
    cout << "vvvvvvvvvvvv"<<endl; 
    vector<float> source_face_embedding = face_embedding_net_trt.detect(source_img, face_landmark_5of68_trt);


      cout << "wwwwwwwwww"<<endl; 

    std::cout << "begin to inswapper_128.onnx: " <<  std::endl;
    SwapFace_trt swap_face_net_trt("inswapper_128.onnx", config);
    std::cout << "inswapper_128.onnx  trted: " <<  std::endl;
    samplesCommon::BufferManager buffers(swap_face_net_trt.m_trtEngine_faceswap->m_engine);
    cout << "inswapper_128 done"<<endl;
    cv::Mat swapimg = swap_face_net_trt.process(target_img, source_face_embedding, target_landmark_5, buffers);
    
    cout << "swap_face_net.process end" <<endl;
    imwrite("swapimg.jpg", swapimg);
    // samplesCommon::Args args; // 接收用户传递参数的变量

    // //SampleOnnxMNIST sample0;//(initializeSampleParams(args)); // 定义一个sample实例
    // SampleOnnxMNIST sample(initializeSampleParams(args)); // 定义一个sample实例

    //  if (sample.build()) // 【主要】在build方法中构建网络，返回构建网络是否成功的状态
    // {
    //     //return sample::gLogger.reportFail(sampleTest);
    //     std::cout << "build() is ok"<<std::endl;
        
    //  }
    //  if (!sample.infer()) // 【主要】读取图像并进行推理，返回推理是否成功的状态
    //  {
    //      std::cout << "infer failed"<<std::endl;
    //      //return sample::gLogger.reportFail(sampleTest);
    //  }





    // if (!arguments.onnxModelPath.empty()) {
    //     // Build the onnx model into a TensorRT engine file, and load the TensorRT
    //     // engine file into memory.
    //     bool succ = engine.buildLoadNetwork(arguments.onnxModelPath, subVals, divVals, normalize);
    //     if (!succ) {
    //         throw std::runtime_error("Unable to build or load TensorRT engine.");
    //     }
    // } else {
    //     // Load the TensorRT engine file directly
    //     bool succ = engine.loadNetwork(arguments.trtModelPath, subVals, divVals, normalize);
    //     if (!succ) {
    //         const std::string msg = "Unable to load TensorRT engine.";
    //         spdlog::error(msg);
    //         throw std::runtime_error(msg);
    //     }
    // }

    // Read the input image
    // TODO: You will need to read the input image required for your model
    // const std::string inputImage = "6.jpg";
    // auto cpuImg = cv::imread(inputImage);
    // if (cpuImg.empty()) {
    //     const std::string msg = "Unable to read image at path: " + inputImage;
    //     spdlog::error(msg);
    //     throw std::runtime_error(msg);
    // }

    // // Upload the image GPU memory
    // cv::cuda::GpuMat img;
    // img.upload(cpuImg);

    // // The model expects RGB input
    // cv::cuda::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // // In the following section we populate the input vectors to later pass for
    // // inference
    // const auto &inputDims = engine.getInputDims();
    // std::vector<std::vector<cv::cuda::GpuMat>> inputs;

    // Let's use a batch size which matches that which we set the
    // Options.optBatchSize option
    //size_t batchSize = options.optBatchSize;

    // TODO:
    // For the sake of the demo, we will be feeding the same image to all the
    // inputs You should populate your inputs appropriately.
    // for (const auto &inputDim : inputDims) { // For each of the model inputs...
    //     std::vector<cv::cuda::GpuMat> input;
    //     for (size_t j = 0; j < batchSize; ++j) { // For each element we want to add to the batch...
    //         // TODO:
    //         // You can choose to resize by scaling, adding padding, or a combination
    //         // of the two in order to maintain the aspect ratio You can use the
    //         // Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while
    //         // maintain the aspect ratio (adds padding where necessary to achieve
    //         // this).
    //         auto resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(img, inputDim.d[1], inputDim.d[2]);
    //         // You could also perform a resize operation without maintaining aspect
    //         // ratio with the use of padding by using the following instead:
    //         //            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2],
    //         //            inputDim.d[1])); // TRT dims are (height, width) whereas
    //         //            OpenCV is (width, height)
    //         input.emplace_back(std::move(resized));
    //     }
    //     inputs.emplace_back(std::move(input));
    // }

    // // Warm up the network before we begin the benchmark
    // spdlog::info("Warming up the network...");
    // std::vector<std::vector<std::vector<float>>> featureVectors;
    // for (int i = 0; i < 100; ++i) {
    //     bool succ = engine.runInference(inputs, featureVectors);
    //     if (!succ) {
    //         const std::string msg = "Unable to run inference.";
    //         spdlog::error(msg);
    //         throw std::runtime_error(msg);
    //     }
    // }

    // Benchmark the inference time
    // size_t numIterations = 1000;
    // spdlog::info("Running benchmarks ({} iterations)...", numIterations);
    // preciseStopwatch stopwatch;
    // for (size_t i = 0; i < numIterations; ++i) {
    //     featureVectors.clear();
    //     engine.runInference(inputs, featureVectors);
    // }
    // auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    // auto avgElapsedTimeMs = totalElapsedTimeMs / numIterations / static_cast<float>(inputs[0].size());

    // spdlog::info("Benchmarking complete!");
    // spdlog::info("======================");
    // spdlog::info("Avg time per sample: ");
    // spdlog::info("Avg time per sample: {} ms", avgElapsedTimeMs);
    // spdlog::info("Batch size: {}", inputs[0].size());
    // spdlog::info("Avg FPS: {} fps", static_cast<int>(1000 / avgElapsedTimeMs));
    // spdlog::info("======================\n");

    // // Print the feature vectors
    // for (size_t batch = 0; batch < featureVectors.size(); ++batch) {
    //     for (size_t outputNum = 0; outputNum < featureVectors[batch].size(); ++outputNum) {
    //         spdlog::info("Batch {}, output {}", batch, outputNum);
    //         std::string output;
    //         int i = 0;
    //         for (const auto &e : featureVectors[batch][outputNum]) {
    //             output += std::to_string(e) + " ";
    //             if (++i == 10) {
    //                 output += "...";
    //                 break;
    //             }
    //         }
    //         spdlog::info("{}", output);
    //     }
    // }

    // TODO: If your model requires post processing (ex. convert feature vector
    // into bounding boxes) then you would do so here.

    return 0;
}
