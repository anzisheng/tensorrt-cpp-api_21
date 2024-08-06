#include"faceenhancer_trt.h"

using namespace cv;
using namespace std;
//using namespace Ort;

FaceEnhance_trt::FaceEnhance_trt(const std::string &onnxModelPath, const YoloV8Config &config, int method)
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
    cout << "m_trtEngine_enhance will be constrect"<<endl;
    // Create our TensorRT inference engine
    m_trtEngine_enhance = std::make_unique<Engine<float>>(options);
    cout << "m_trtEngine_enhance have constrected"<<endl;


    auto succ = m_trtEngine_enhance->buildLoadNetwork(onnxModelPath, SUB_VALS, DIV_VALS, NORMALIZE,1);
    if (!succ) {
        const std::string errMsg = "Error: Unable to build or load the TensorRT engine. "
                                   "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp).";
        throw std::runtime_error(errMsg);
    }

     ////OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///如果使用cuda加速，需要取消注释

    //sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    /// std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  ////windows写法
    /// ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows写法
    //ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法
    /*
    size_t numInputNodes = ort_session->GetInputCount();
    size_t numOutputNodes = ort_session->GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < numInputNodes; i++)
    {
	char *mem;
        //input_names.push_back(ort_session->GetInputName(i, allocator)); /// 低版本onnxruntime的接口函数
        AllocatedStringPtr input_name_Ptr = ort_session->GetInputNameAllocated(i, allocator);  /// 高版本onnxruntime的接口函数
	mem = (char *) malloc(strlen(input_name_Ptr.get()));
	memcpy(mem, input_name_Ptr.get(), strlen(input_name_Ptr.get()) + 1);
        input_names.push_back(mem); /// 高版本onnxruntime的接口函数
        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }
    for (int i = 0; i < numOutputNodes; i++)
    {
	char *mem;
        //output_names.push_back(ort_session->GetOutputName(i, allocator)); /// 低版本onnxruntime的接口函数
        AllocatedStringPtr output_name_Ptr= ort_session->GetOutputNameAllocated(i, allocator);
	mem = (char *) malloc(strlen(output_name_Ptr.get()));
	std::memcpy(mem, output_name_Ptr.get(), strlen(output_name_Ptr.get()) + 1);
        output_names.push_back(mem); /// 高版本onnxruntime的接口函数
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }*/

    //this->input_height = input_node_dims[0][2];
    //this->input_width = input_node_dims[0][3];

    ////在这里就直接定义了，没有像python程序里的那样normed_template = TEMPLATES.get(template) * crop_size
    this->normed_template.emplace_back(Point2f(192.98138112, 239.94707968));
    this->normed_template.emplace_back(Point2f(318.90276864, 240.19360256));
    this->normed_template.emplace_back(Point2f(256.63415808, 314.01934848));
    this->normed_template.emplace_back(Point2f(201.26116864, 371.410432));
    this->normed_template.emplace_back(Point2f(313.0890496,  371.1511808));
}
/*
std::vector<std::vector<cv::cuda::GpuMat>> FaceEnhance_trt::preprocess(const cv::cuda::GpuMat &gpuImg) 
{
    // Populate the input vectors
    cout << "m_trtEngine_enhance:"<<endl;//<< m_trtEngine_enhance<<endl;;
    

    const auto &inputDims = m_trtEngine_enhance->getInputDims();
    cout << "inputDims[0].d[1] :" << inputDims[0].d[1]<<endl;

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
    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    
    std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};

    // These params will be used in the post-processing stage
    //m_imgHeight = rgbMat.rows;
    //m_imgWidth = rgbMat.cols;
    //m_ratio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));


    return inputs;

}
*/

void FaceEnhance_trt::preprocess(Mat srcimg, const vector<Point2f> face_landmark_5, Mat& affine_matrix, Mat& box_mask, samplesCommon::BufferManager &buffers)
{
    const int inputH = 512;//mInputDims.d[2];
    const int inputW = 512;//mInputDims.d[3];

    Mat crop_img;
    affine_matrix = warp_face_by_face_landmark_5(srcimg, crop_img, face_landmark_5, this->normed_template, Size(512, 512));
    //imwrite("endhance_crop_last.jpg", crop_img);
    const int crop_size[2] = {crop_img.cols, crop_img.rows};
    box_mask = create_static_box_mask(crop_size, this->FACE_MASK_BLUR, this->FACE_MASK_PADDING);
     
    //std::cout << "crop_img: "<< crop_img.rows <<std::endl;
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

    //std::cout << "1111 " << endl;
    float* hostDataBuffer0 = static_cast<float*>(buffers.getHostBuffer("input"));//static_cast<float*>(buffers.mManagedBuffers[0]->hostBuffer.data());
    // for (int i = 0; i < 512 * 512*3; i++)
    // {
    //     hostDataBuffer0[i] = input_image[i];

    // }
    memcpy(hostDataBuffer0, input_image.data(), single_chn_size*3);


 /*

    cout << "22222222222222" <<endl;

     vector<cv::Mat> bgrChannels(3);
    split(crop_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / (255.0*0.5), -1.0);
    }

    const int image_area = 512 * 512;
    this->input_image.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image.data(), (float *)bgrChannels[2].data, single_chn_size);    ///rgb顺序
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[0].data, single_chn_size);
    
    const int inputH = 512;//mInputDims.d[2];
    const int inputW = 512;//mInputDims.d[3];
    cout <<"target host"<<endl;
    float* hostDataBuffer0 = static_cast<float*>(buffers.getHostBuffer("input"));//static_cast<float*>(buffers.mManagedBuffers[0]->hostBuffer.data());
    cout <<"target host pointer"<<endl;

    memcpy(hostDataBuffer0, this->input_image.data(), 3*image_area * sizeof(float));
    cout << "input000000000000" <<endl;
    */

}
/*

Mat FaceEnhance_trt::process(cv::cuda::GpuMat gpuImbBGR, const vector<Point2f> target_landmark_5)
{
    cout << "preprocessing" <<endl;
    const auto inputs = preprocess(gpuImbBGR);

    cout << "call enhance runInference"<<endl;    
    std::vector<std::vector<std::vector<float>>> featureVectors;
    auto succ = m_trtEngine_enhance->runInference(inputs, featureVectors);
    cout << "next to post processing"<<endl;

     cv::Mat ret;
     //virtual const std::vector<nvinfer1::Dims> &getOutputDims() const = 0;
     const auto &numOutputs = m_trtEngine_enhance->getOutputDims().size();
     const auto output = m_trtEngine_enhance->getOutputDims()[0];
     //cout << "numOutputs: " <<numOutputs[0]<<"  "<<numOutputs[0].d[1]<< "  "numOutputs[0].d[2]<<"  "<<numOutputs[0].d[3]<<endl;
     cout << "output dims: " <<output.d[0]<<"  "<<output.d[1]<< "  "<<output.d[2]<<"  "<<output.d[3]<<endl;
    //  const auto &outputDims = m_trtEngine->getOutputDims();
    // auto numChannels = outputDims[0].d[1];
    // auto numAnchors = outputDims[0].d[2];
     
     if (numOutputs == 1) {
        // Object detection or pose estimation
        // Since we have a batch size of 1 and only 1 output, we must convert the output from a 3D array to a 1D array.
        std::vector<float> featureVector;
        Engine<float>::transformOutput(featureVectors, featureVector);

        const auto &outputDims = m_trtEngine_enhance->getOutputDims();
        int numChannels = outputDims[outputDims.size() - 1].d[1];
        cout << "numChannels: "<< numChannels<<endl;

        //float* rdata = featureVector.data();
        float* pdata = featureVector.data();
        vector<float> result_data(512*512*3);        
        //cudaMemcpy(result_data.data(), pdata, 512*512*3*sizeof(float), cudaMemcpyHostToHost); //import not cudaMemcpyDeviceToHost

        //float* pdata = featureVector.data();//ort_outputs[0].GetTensorMutableData<float>();

        // for (int i = 30000 - 100; i< 30000; i++)
        //         cout << pdata[i]<<endl;

        std::vector<int64_t> outs_shape = {1, 3, 512,512};//ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const int out_h = outs_shape[2];
        cout << "post result :"<< outs_shape[0] << " "<< outs_shape[1] << " "<<outs_shape[2] << " "<<outs_shape[3];
        const int out_w = outs_shape[3];
        const int channel_step = out_h * out_w;
        Mat rmat(out_h, out_w, CV_32FC1, pdata);
        imwrite("red.jpg", rmat);
        Mat gmat(out_h, out_w, CV_32FC1, pdata + channel_step);
        imwrite("green.jpg", gmat);
        Mat bmat(out_h, out_w, CV_32FC1, pdata + 2 * channel_step);
        imwrite("blue.jpg", bmat);
        rmat.setTo(-1, rmat < -1);
        rmat.setTo(1, rmat > 1);
        rmat = (rmat+1)*0.5;
        gmat.setTo(-1, gmat < -1);
        gmat.setTo(1, gmat > 1);
        gmat = (gmat+1)*0.5;
        bmat.setTo(-1, bmat < -1);
        bmat.setTo(1, bmat > 1);
        bmat = (bmat+1)*0.5;

        rmat *= 255.f;
        gmat *= 255.f;
        bmat *= 255.f;
        rmat.setTo(0, rmat < 0);
        rmat.setTo(255, rmat > 255);
        gmat.setTo(0, gmat < 0);
        gmat.setTo(255, gmat > 255);
        bmat.setTo(0, bmat < 0);
        bmat.setTo(255, bmat > 255);

        vector<Mat> channel_mats(3);
        channel_mats[0] = bmat;
        channel_mats[1] = gmat;
        channel_mats[2] = rmat;
        Mat result;
        merge(channel_mats, result);
        cout <<"merged"<<endl;
        imwrite("merge_result.jpg", result);
        result.convertTo(result, CV_8UC3);
        imwrite("convert_result.jpg", result);
        cout <<"convered"<<endl;
        // TODO: Need to improve this to make it more generic (don't use magic number).
        // For now it works with Ultralytics pretrained models.
        // if (numChannels == 56) {
        //     // Pose estimation
        //     ret = postprocessPose(featureVector);
        // } else {
        //     // Object detection
            //ret = postprocessDetect(featureVector);
        //}



    }

   
    return ret;
    

}*/

cv::Mat FaceEnhance_trt::verifyOutput(Mat &target_img, const samplesCommon::BufferManager& buffers, Mat& affine_matrix, Mat& box_mask)
{
    //const int outputSize = 512;
    //cout << "verifyOutput....."<< target_img.rows<<"  "<<target_img.cols <<endl;
    //imwrite("strange.jpg", target_img);
    
    float* output = static_cast<float*>(buffers.getHostBuffer("output")); // "output"
    std::vector<float> vdata;
    vdata.resize(512*512*3);
    float* pdata = vdata.data();
    // for(int i = 0; i<512*512*3; i++ )
    // {
    //     vdata[i] = *(output+i);        
    //     std::cout << vdata[i]<<std::endl;
    // }
    memcpy(pdata, output, 512*512*3*sizeof(float));
    const int out_h = 512;//outs_shape[2];
	const int out_w = 512;//outs_shape[3];
	const int channel_step = out_h * out_w;
	Mat rmat(out_h, out_w, CV_32FC1, pdata);
	Mat gmat(out_h, out_w, CV_32FC1, pdata + channel_step);
	Mat bmat(out_h, out_w, CV_32FC1, pdata + 2 * channel_step);
    
    
	rmat *= 255.f;    
    gmat *= 255.f;
	bmat *= 255.f;
    
    rmat.setTo(0, rmat < 0);
	rmat.setTo(255, rmat > 255);
	gmat.setTo(0, gmat < 0);
    
	gmat.setTo(255, gmat > 255);
	bmat.setTo(0, bmat < 0);
	bmat.setTo(255, bmat > 255);
    
	vector<Mat> channel_mats(3);
	channel_mats[0] = bmat;
	channel_mats[1] = gmat;
	channel_mats[2] = rmat;
    Mat result;
    //cout << "begin merge" <<endl;

	// merge(channel_mats, result);
    // imwrite("result.jpg", result);

    // cout << "after merge" <<endl;
    //Mat result;
	merge(channel_mats, result);
	result.convertTo(result, CV_8UC3);

    box_mask.setTo(0, box_mask < 0);
	box_mask.setTo(1, box_mask > 1);
    Mat paste_frame = paste_back(target_img, result, box_mask, affine_matrix);
    Mat dstimg = blend_frame(target_img, paste_frame);
    cout << "done" <<endl;
    return dstimg;



    
    //return result;
    /*cout << "enhance output"<<endl;
    // cv::Mat crop_img  = cv::imread("box_mask.jpg");
    // std::cout << "crop_img: "<< crop_img.rows <<std::endl;
    //const int outputSize = 512*512*3;
    //cout << "mParams.outputTensorNames[0]:"<< mParams.outputTensorNames[0]<<endl;
    float* pdata = static_cast<float*>(buffers.getHostBuffer("output"));//("output")); //"output" //mParams.outputTensorNames[0]
    //std::vector<int64_t> outs_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	const int out_h = 512;//outs_shape[2];
    //cout << "post result :"<< outs_shape[0] << " "<< outs_shape[1] << " "<<outs_shape[2] << " "<<outs_shape[3];
	const int out_w = 512;//outs_shape[3];
	const int channel_step = out_h * out_w;
    cout << "enhance output0000"<<endl;
	Mat rmat(out_h, out_w, CV_32FC1, pdata);
    cout << "enhance output1111"<<endl;
    imwrite("red.jpg", rmat);
	Mat gmat(out_h, out_w, CV_32FC1, pdata + channel_step);
    imwrite("green.jpg", gmat);
	Mat bmat(out_h, out_w, CV_32FC1, pdata + 2 * channel_step);
    imwrite("blue.jpg", bmat);
    cout << "enhance output2222"<<endl;
    rmat.setTo(-1, rmat < -1);
    cout << "enhance output333"<<endl;
	rmat.setTo(1, rmat > 1);
    rmat = (rmat+1)*0.5;
	gmat.setTo(-1, gmat < -1);
	gmat.setTo(1, gmat > 1);
    gmat = (gmat+1)*0.5;
	bmat.setTo(-1, bmat < -1);
	bmat.setTo(1, bmat > 1);
    bmat = (bmat+1)*0.5;
    cout << "enhance outpu44444"<<endl;

    rmat *= 255.f;
	gmat *= 255.f;
	bmat *= 255.f;
    rmat.setTo(0, rmat < 0);
	rmat.setTo(255, rmat > 255);
	gmat.setTo(0, gmat < 0);
	gmat.setTo(255, gmat > 255);
	bmat.setTo(0, bmat < 0);
	bmat.setTo(255, bmat > 255);
 cout << "enhance outpu5555"<<endl;
	vector<Mat> channel_mats(3);
	channel_mats[0] = bmat;
	channel_mats[1] = gmat;
	channel_mats[2] = rmat;
     cout << "enhance outpu6666"<<endl;
    Mat result;
	merge(channel_mats, result);
     cout << "enhance outpu7777"<<endl;
    //cout <<""<<endl;
    imwrite("merge_result.jpg", result);
	result.convertTo(result, CV_8UC3);
    imwrite("convert_result.jpg", result);
     cout << "enhance outpu8888"<<endl;

    box_mask.setTo(0, box_mask < 0);
	box_mask.setTo(1, box_mask > 1);
    Mat paste_frame = paste_back(target_img, result, box_mask, affine_matrix);
     cout << "enhance outpu9999"<<endl;
    Mat dstimg = blend_frame(target_img, paste_frame);
    cout << "enhance outpu*****"<<endl;*/
    //return dstimg;
}

Mat FaceEnhance_trt::process(Mat target_img, const vector<Point2f> target_landmark_5, samplesCommon::BufferManager &buffers)
{
    Mat affine_matrix;
    Mat box_mask;
    //cout << "going enhance preprocess"<<endl;

    this->preprocess(target_img, target_landmark_5, affine_matrix, box_mask, buffers);
    //cout << "going copyInputToDevice"<<endl;
    buffers.copyInputToDevice();
    //cout << "going FaceEnhance_trt cexecuteV2"<<endl;

    m_trtEngine_enhance->m_context->executeV2(buffers.getDeviceBindings().data());

     // Memcpy from device output buffers to host output buffers
    
    buffers.copyOutputToHost();
    //cout << "going FaceEnhance_trt verifyOutput"<<endl;
    return verifyOutput(target_img, buffers, affine_matrix,  box_mask);


    /*
    Mat crop_img;
    affine_matrix = warp_face_by_face_landmark_5(target_img, crop_img, target_landmark_5, this->normed_template, Size(512, 512));
    imwrite("enhance_crop.jpg", crop_img);
    const int crop_size[2] = {crop_img.cols, crop_img.rows};
    box_mask = create_static_box_mask(crop_size, this->FACE_MASK_BLUR, this->FACE_MASK_PADDING);
    imwrite("enhance_box_mask.jpg", box_mask);
    cout << "before preprocess"<<endl;

    cv::cuda::GpuMat gpuImbBGR;
    gpuImbBGR.upload(target_img);

    process(gpuImbBGR, target_landmark_5);
    */
    //this->preprocess(target_img, target_landmark_5);


    // std::vector<int64_t> input_img_shape = {1, 3, this->input_height, this->input_width};
    // Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());

    // Ort::RunOptions runOptions;
    // vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, 1, this->output_names.data(), output_names.size());

    // float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    // std::vector<int64_t> outs_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	// const int out_h = outs_shape[2];
	// const int out_w = outs_shape[3];
	// const int channel_step = out_h * out_w;
	// Mat rmat(out_h, out_w, CV_32FC1, pdata);
	// Mat gmat(out_h, out_w, CV_32FC1, pdata + channel_step);
	// Mat bmat(out_h, out_w, CV_32FC1, pdata + 2 * channel_step);
    // rmat.setTo(-1, rmat < -1);
	// rmat.setTo(1, rmat > 1);
    // rmat = (rmat+1)*0.5;
	// gmat.setTo(-1, gmat < -1);
	// gmat.setTo(1, gmat > 1);
    // gmat = (gmat+1)*0.5;
	// bmat.setTo(-1, bmat < -1);
	// bmat.setTo(1, bmat > 1);
    // bmat = (bmat+1)*0.5;

    // rmat *= 255.f;
	// gmat *= 255.f;
	// bmat *= 255.f;
    // rmat.setTo(0, rmat < 0);
	// rmat.setTo(255, rmat > 255);
	// gmat.setTo(0, gmat < 0);
	// gmat.setTo(255, gmat > 255);
	// bmat.setTo(0, bmat < 0);
	// bmat.setTo(255, bmat > 255);

	// vector<Mat> channel_mats(3);
	// channel_mats[0] = bmat;
	// channel_mats[1] = gmat;
	// channel_mats[2] = rmat;
    // Mat result;
	// merge(channel_mats, result);
	// result.convertTo(result, CV_8UC3);

    // box_mask.setTo(0, box_mask < 0);
	// box_mask.setTo(1, box_mask > 1);
    // Mat paste_frame = paste_back(target_img, result, box_mask, affine_matrix);
     //Mat dstimg ;//= blend_frame(target_img, paste_frame);
    //return dstimg;
}

void FaceEnhance_trt::memoryFree() {
}


