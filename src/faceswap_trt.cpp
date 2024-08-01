#include"faceswap_trt.h"

using namespace cv;
using namespace std;
//using namespace Ort;

SwapFace_trt::SwapFace_trt(string model_path, const YoloV8Config &config)
{
    //m_buffers = samplesCommon::BufferManager(m_trtEngine_faceswap->m_engine);

    ////OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///如果使用cuda加速，需要取消注释
/*
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    /// std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  ////windows写法
    /// ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows写法
    ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法
    
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
    }
    */
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
    m_trtEngine_faceswap = std::make_unique<Engine<float>>(options);

// Build the onnx model into a TensorRT engine file, cache the file to disk, and then load the TensorRT engine file into memory.
    // If the engine file already exists on disk, this function will not rebuild but only load into memory.
    // The engine file is rebuilt any time the above Options are changed.
    auto succ = m_trtEngine_faceswap->buildLoadNetwork(model_path, SUB_VALS, DIV_VALS, NORMALIZE);
    if (!succ) {
        const std::string errMsg = "Error: Unable to build or load the TensorRT engine. "
                                   "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp).";
        throw std::runtime_error(errMsg);
    }


    // this->input_height = input_node_dims[0][2];
    // this->input_width = input_node_dims[0][3];
    
    const int length = this->len_feature*this->len_feature;
    this->model_matrix = new float[length];
    //cout<<"start read model_matrix.bin"<<endl;
    FILE* fp = fopen("./model_matrix.bin", "rb");
    size_t ret = fread(this->model_matrix, sizeof(float), length, fp);//导入数据
    if (ret) {}
    fclose(fp);//关闭文件
    //cout<<"read model_matrix.bin finish"<<endl;

    ////在这里就直接定义了，没有像python程序里的那样normed_template = TEMPLATES.get(template) * crop_size
    this->normed_template.emplace_back(Point2f(46.29459968, 51.69629952));
    this->normed_template.emplace_back(Point2f(81.53180032, 51.50140032));
    this->normed_template.emplace_back(Point2f(64.02519936, 71.73660032));
    this->normed_template.emplace_back(Point2f(49.54930048, 92.36550016));
    this->normed_template.emplace_back(Point2f(78.72989952, 92.20409984));

    //cout<<"normed_template assign values"<<endl;

}

SwapFace_trt::~SwapFace_trt()
{
	delete[] this->model_matrix;
	this->model_matrix = nullptr;
    this->normed_template.clear();
}

void SwapFace_trt::preprocess(Mat srcimg, const vector<Point2f> face_landmark_5, const vector<float> source_face_embedding, 
Mat& affine_matrix, Mat& box_mask, samplesCommon::BufferManager &buffers)
{
    Mat crop_img;
    cout << "10101010" <<endl;
    affine_matrix = warp_face_by_face_landmark_5(srcimg, crop_img, face_landmark_5, this->normed_template, Size(128, 128));
    cout << "44444444444" <<endl;
    //imwrite("swap_crop.jpg", crop_img);
    const int crop_size[2] = {crop_img.cols, crop_img.rows};
    box_mask = create_static_box_mask(crop_size, this->FACE_MASK_BLUR, this->FACE_MASK_PADDING);
    //crop_img = imread("swap_face_crop_img.jpg");
    vector<cv::Mat> bgrChannels(3);
    split(crop_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        //bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / (255.0*this->INSWAPPER_128_MODEL_STD[c]), -this->INSWAPPER_128_MODEL_MEAN[c]/this->INSWAPPER_128_MODEL_STD[c]);
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / (255.0*this->INSWAPPER_128_MODEL_STD[c]), -this->INSWAPPER_128_MODEL_MEAN[c]/this->INSWAPPER_128_MODEL_STD[c]);
    }

    const int image_area = 128 * 128;
    //cout << "22222222222222" <<endl;
    this->input_image.resize(3 * image_area);
    //cout << "3333333" <<endl;
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image.data(), (float *)bgrChannels[2].data, single_chn_size);    ///rgb顺序
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[0].data, single_chn_size);

    const int inputH = 128;//mInputDims.d[2];
    const int inputW = 128;//mInputDims.d[3];
    cout <<"target host"<<endl;
    float* hostDataBuffer0 = static_cast<float*>(buffers.getHostBuffer("target"));//static_cast<float*>(buffers.mManagedBuffers[0]->hostBuffer.data());
    cout <<"target host pointer"<<endl;
    // for (int i = 0; i < 128*128*3; i++)
    // {
    //     hostDataBuffer0[i] = input_image[i];

    // }
    memcpy(hostDataBuffer0, this->input_image.data(), 3*image_area * sizeof(float));
    //read target from file
    // fstream target_File("target_input.txt", ios::in); 
    // if(!target_File.is_open())
    // {
    //     cout << "cann't open the wapface_output.txt"<<endl;
    // }
    // for (int i = 0; i < 128*128*3; i++)
    // {
    //     float x; target_File >> x;
    //     //vdata[i] = x;
    //     hostDataBuffer0[i]= x;
    // }
    // target_File.close();
    





    float linalg_norm = 0;
    for(int i=0;i<this->len_feature;i++)
    {
        linalg_norm += powf(source_face_embedding[i], 2);
    }
    linalg_norm = sqrt(linalg_norm);
    this->input_embedding.resize(this->len_feature);
    for(int i=0;i<this->len_feature;i++)
    {
        float sum=0;
        for(int j=0;j<this->len_feature;j++)
        {
            sum += (source_face_embedding[j]*this->model_matrix[j*this->len_feature+i]);
        }
        this->input_embedding[i] = sum/linalg_norm;
    }
    #ifdef SHOW
    cout << "this->input_embedding show:"<<endl;
    for(int i = 0; i < this->input_embedding.size(); i++ )
    {
        //cout << i << ": "<< this->input_embedding[i] << std::endl;
    }
    // ifstream srcFile("source_input.txt", ios::in);
    // if(!srcFile.is_open())
    // {
    //     cout << "cann't open the input_embeddingt.txt"<<endl;
    // }
    cout << "reading in data from input_embeddingt.txt"<<endl;
    // for(int i = 0; i < this->input_embedding.size(); i++ )
    // {
    //     float x;
    //     srcFile >> x;
    //     this->input_embedding[i] = x;
    // }
    #endif

    float* hostDataBuffer1 = static_cast<float*>(buffers.getHostBuffer("source"));
    memcpy(hostDataBuffer1, this->input_embedding.data(), this->len_feature * sizeof(float));
    // ifstream source_File("source_input.txt", ios::in);
    // for(int i = 0; i < this->input_embedding.size(); i++ )
    // {
    //     float x;
    //     source_File >> x;
    //     //this->input_embedding[i] = x;
    //     hostDataBuffer1[i] = x;
    // }
    // source_File.close();

}

Mat SwapFace_trt::process(Mat target_img, const vector<float> source_face_embedding, const vector<Point2f> target_landmark_5,samplesCommon::BufferManager &buffers)
{
    Mat affine_matrix;
    Mat box_mask;
    cout << "going preprocess"<<endl;
    this->preprocess(target_img, target_landmark_5, source_face_embedding, affine_matrix, box_mask, buffers);
    buffers.copyInputToDevice();

    m_trtEngine_faceswap->m_context->executeV2(buffers.getDeviceBindings().data());

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    return verifyOutput(target_img, buffers, affine_matrix,  box_mask);




    // std::vector<Ort::Value> inputs_tensor;
    // std::vector<int64_t> input_img_shape = {1, 3, this->input_height, this->input_width};
    // inputs_tensor.emplace_back(Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size()));
    // std::vector<int64_t> input_embedding_shape = {1, this->len_feature};
    // inputs_tensor.emplace_back(Value::CreateTensor<float>(memory_info_handler, this->input_embedding.data(), this->input_embedding.size(), input_embedding_shape.data(), input_embedding_shape.size()));
    

    // Ort::RunOptions runOptions;
    // vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), inputs_tensor.data(), inputs_tensor.size(), this->output_names.data(), output_names.size());

    // float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    // std::vector<int64_t> outs_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	// const int out_h = outs_shape[2];
	// const int out_w = outs_shape[3];
	// const int channel_step = out_h * out_w;
	// Mat rmat(out_h, out_w, CV_32FC1, pdata);
	// Mat gmat(out_h, out_w, CV_32FC1, pdata + channel_step);
	// Mat bmat(out_h, out_w, CV_32FC1, pdata + 2 * channel_step);
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

    // box_mask.setTo(0, box_mask < 0);
	// box_mask.setTo(1, box_mask > 1);
    // Mat dstimg = paste_back(target_img, result, box_mask, affine_matrix);
    // return dstimg;
}
cv::Mat SwapFace_trt::verifyOutput(Mat &target_img, const samplesCommon::BufferManager& buffers, Mat& affine_matrix, Mat& box_mask)
{
    //const int outputSize = 128*128*3;
    //cout << "mParams.outputTensorNames[0]:"<< mParams.outputTensorNames[0]<<endl;
    float* output = static_cast<float*>(buffers.getHostBuffer("output")); //"output" //mParams.outputTensorNames[0]
    // ofstream destFile2("swapface_output.txt", ios::out); 
    // cout << "embedding show:"<<endl;
	// for(int i =0; i < 128*128*3; i++)
	// {
	//  	destFile2 << *(output+i) << " " ;
    //     //cout << source_face_embedding[i] << " ";
	// }
	// destFile2.close();
    // fstream target_File("swapface_output_mnist.txt", ios::in); 
    // if(!target_File.is_open())
    // {
    //     cout << "cann't open the wapface_output.txt"<<endl;
    // }
    // for (int i = 0; i < 128*128*3; i++)
    // {
    //     float x; target_File >> x;
    //     //vdata[i] = x;
    //     //hostDataBuffer0[i]= x;
    //     output[i]= x;
    // }
    // target_File.close();

    // std::vector<float> vdata(128*128*3);
    // fstream srcFile("swapface_output_cpu.txt", ios::in);
    // if(!srcFile.is_open())
    // {
    // cout << "cann't open the swapface_output_cpu.txt"<<endl;
    // }
    // for (int i = 0; i < 128*128*3; i++)
    // {
    // float x; srcFile >> x;
    // vdata[i] = x;
    // }
    float* pdata =  output;//vdata.data();//output;//


    // std::vector<float> vdata;
    // vdata.resize(outputSize);
    // float* pdata = vdata.data();
    // for(int i = 0; i<128*128*3; i++ )
    // {
    //     vdata[i] = *(output+i);        
    //     //std::cout << vdata[i]<<std::endl;
    // }
    
    //memcpy(pdata, output, 128*128*3*sizeof(float));
    const int out_h = 128;//outs_shape[2];
	const int out_w = 128;//outs_shape[3];
	const int channel_step = out_h * out_w;
	/*Mat rmat(out_h, out_w, CV_32FC1, pdata);
	Mat gmat(out_h, out_w, CV_32FC1, pdata + channel_step);
	Mat bmat(out_h, out_w, CV_32FC1, pdata + 2 * channel_step);
    std::cout << "********* " << endl;
    std::cout << rmat.rows <<"  "<< rmat.cols <<endl;
	rmat *= 255.f;
    std::cout << "&&&&& " << endl;
    gmat *= 255.f;
	bmat *= 255.f;
    std::cout << "aaaaaaaa " << endl;
    rmat.setTo(0, rmat < 0);
	rmat.setTo(255, rmat > 255);
	gmat.setTo(0, gmat < 0);
    std::cout << "bbbbbbbbbb " << endl;
	gmat.setTo(255, gmat > 255);
	bmat.setTo(0, bmat < 0);
	bmat.setTo(255, bmat > 255);
    std::cout << "cccccccc " << endl;
	vector<Mat> channel_mats(3);
	channel_mats[0] = bmat;
	channel_mats[1] = gmat;
	channel_mats[2] = rmat;
    Mat result;
     std::cout << "********* " << endl;
     std::cout << "********* " << endl;
	merge(channel_mats, result);
    std::cout << "********* " << endl;
    imwrite("result.jpg", result);

    std::cout << "*++++++++++ " << endl;
    box_mask.setTo(0, box_mask < 0);
	box_mask.setTo(1, box_mask > 1);
    Mat dstimg = paste_back(target_img, result, box_mask, affine_matrix);
    imwrite("result----.jpg", dstimg);
    return dstimg;*/

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
	merge(channel_mats, result);
    //box_mask = imread("faceswap_box_mask.jpg"); //anzisheng
    //cout << affine_matrix <<endl;

    box_mask.setTo(0, box_mask < 0);
	box_mask.setTo(1, box_mask > 1);
    Mat dstimg = paste_back(target_img, result, box_mask, affine_matrix);
    imwrite("result99999.jpg", dstimg);
    return dstimg;


}

// void SwapFace_trt::memoryFree() {
//     for (auto it = this->input_names.begin(); it != this->input_names.end(); ++it) {
//         free(*it);
//     }

//     for (auto it = this->output_names.begin(); it != this->output_names.end(); ++it) {
//         free(*it);
//     }
// }
