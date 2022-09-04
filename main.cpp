#include<opencv2/highgui.hpp>
#include<iostream>
#include<opencv2/imgproc.hpp>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/text/ocr.hpp>
#include <string>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <algorithm>
using namespace std;
using namespace cv::dnn;
using namespace cv;
constexpr float CONFIDENCE_THRESHOLD = 0.8;
constexpr float NMS_THRESHOLD = 0.5;
constexpr int NUM_CLASSES = 1;
// Constants.
const float INPUT_WIDTH = 320.0;
const float INPUT_HEIGHT = 320.0;
const float SCORE_THRESHOLD = 0.5;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);

// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
Mat detect_text(Mat large){
    Mat rgb;
    Mat text_detect;
    // downsample and use it for processing
    pyrDown(large, rgb);
    Mat small;
    cvtColor(rgb, small, cv::COLOR_BGR2GRAY);
    // morphological gradient
    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);
    // binarize
    Mat bw;
    threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
    // connect horizontally oriented regions
    Mat connected;
    morphKernel = getStructuringElement(MORPH_RECT, Size(10, 10));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
    // find contours
    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(connected, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE, Point(0, 0));
    // filter contours
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0]){
        Rect rect = boundingRect(contours[idx]);

        Mat maskROI(mask, rect);
        maskROI = Scalar(0, 0, 0);
        // fill the contour
        drawContours(mask, contours, idx, Scalar(255, 255, 255), cv::FILLED);
        // ratio of non-zero pixels in the filled region
        double r = (double)countNonZero(maskROI) / (rect.width * rect.height);
        
        // assume at least 45% of the area is filled if it contains text
        if (r > 0.45 && 
        (rect.height > 8 && rect.width > 8) // constraints on region size
        // these two conditions alone are not very robust. better to use something 
        //like the number of significant peaks in a horizontal projection as a third condition
        ){ 
            // cout<<rect.width<<" "<<rect.height<<endl;
            // cout<<rect.x<<" "<<rect.y<<endl;
            // rectangle(rgb, rect, Scalar(0, 255, 0), 2);
            text_detect=rgb(Rect(rect.x,rect.y,rect.width,rect.height));
        }
    }
    return text_detect;
}
Mat transform(Mat img){
cv::Point2f a1(25, 18), b1(279, 27), c1(279, 79), a2(0, 0), b2(img.size().width, 0), c2(img.size().width, img.size().height);
//cv::Point2f a1(0, 16), b1(303, 28), c1(303, 81), a2(0, 0), b2(img.size().width, 0), c2(img.size().width, img.size().height);
cv::Point2f src[] = {a1, b1, c1};
cv::Point2f dst[] = {a2, b2, c2};
cv::Mat warpMat = cv::getAffineTransform(src, dst);
cv::warpAffine(img, img, warpMat, img.size());
return warpMat;
}
int main()
{   Mat crop;
    tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
    ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
    ocr->SetPageSegMode(tesseract:: PSM_SINGLE_BLOCK  );
    std::vector<std::string> class_names;
    {
        std::ifstream class_file("../classes.txt");
        if (!class_file)
        {
            std::cerr << "failed to open classes.txt\n";
            return 0;
        }
        std::string line;
        while (std::getline(class_file, line))
            class_names.push_back(line);
    }
    // Mat frame;
    std::string image_path = samples::findFile("../L1_Lpn_20220828002713830.jpg");
    Mat frame = imread(image_path, IMREAD_COLOR);
    if(frame.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    VideoCapture cap("../video.mp4");
    if(!cap.isOpened()){
        std::cout << "Could not open the video : " <<endl;
        return -1;
    }
    auto net = cv::dnn::readNetFromDarknet("../weights/yolov4-tiny.cfg", "../weights/yolov4-tiny_final.weights");
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    net.setPreferableBackend(cv::dnn::  DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn:: DNN_TARGET_CPU );
    std::cout<<"Loaded weights successfully..."<<std::endl;
    auto output_names = net.getUnconnectedOutLayersNames();
    cv::Mat  blob;
    std::vector<cv::Mat> detections;
    auto total_start = std::chrono::steady_clock::now();
    while(1){
        cap>>frame;
        if (frame.empty())
        break;
    resize(frame,frame,Size(1280,720),INTER_LINEAR);
    cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(416, 416), cv::Scalar(), true, false, CV_32F);
    net.setInput(blob);
    auto dnn_start = std::chrono::steady_clock::now();
    net.forward(detections, output_names);
    auto dnn_end = std::chrono::steady_clock::now();
    std::vector<int> indices[NUM_CLASSES];
    std::vector<cv::Rect> boxes[NUM_CLASSES];
    std::vector<float> scores[NUM_CLASSES];
    for (auto& output : detections)
    {
        const auto num_boxes = output.rows;
        for (int i = 0; i < num_boxes; i++)
        {
            auto x = output.at<float>(i, 0) * frame.cols;
            auto y = output.at<float>(i, 1) * frame.rows;
            auto width = output.at<float>(i, 2) * frame.cols;
            auto height = output.at<float>(i, 3) * frame.rows;
            cv::Rect rect(x - width/2, y - height/2, width, height);
            for (int c = 0; c < NUM_CLASSES; c++)
            {
                auto confidence = *output.ptr<float>(i, 5 + c);
                if (confidence >= CONFIDENCE_THRESHOLD)
                {
                    boxes[c].push_back(rect);
                    scores[c].push_back(confidence);
                }
            }
        }
    }
    int w,h;
    for (int c = 0; c < NUM_CLASSES; c++)
        cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);
    for (int c= 0; c < NUM_CLASSES; c++)
    {
        for (size_t i = 0; i < indices[c].size(); ++i)
        {  const auto NUM_COLORS = sizeof(colors)/sizeof(colors[0]);
            const auto color = colors[c % NUM_COLORS];
            auto idx = indices[c][i];
            const auto& rect = boxes[c][idx];
            Mat plate_cop;
            frame.copyTo(plate_cop);
          
            // imwrite("plate.png", plate);
            std::ostringstream label_ss;
            label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
            auto label = label_ss.str();
          
            // cout<<"rect.y = "<< rect.y << std::endl;
            // cout<<"rect.x = "<< rect.x << std::endl;
            // cout<<"rect.width = "<< rect.width << std::endl;
            // cout<<"rect.height = "<< rect.height << std::endl;

            int baseline;
            if (scores[c][idx]>0.98 && rect.width>0 && rect.height>0 &&rect.width+rect.x<=frame.cols&&rect.height+rect.y<=frame.rows && rect.x>0 &&rect.y>0){
            Mat plate=frame(Rect(rect.x, rect.y, rect.width, rect.height));
            cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);
            if(rect.width/rect.height <2.5&&rect.width>0 && rect.height>0 &&rect.width+rect.x<=frame.cols&&rect.height+rect.y<=frame.rows && rect.x>0 &&rect.y>0){
                cout<<"Liscen_plate 2 lines"<<std::endl;
                //tach 2 dong
                //dong1
                Mat crop_plate = plate_cop(Rect(rect.x,rect.y,rect.width,(int)(rect.height/2)));
                // Mat crop_plate_tran=transform(crop_plate);
                Mat line_1=detect_text(crop_plate);
                Mat bw,gray;
                if(!line_1.empty())
                    {cvtColor(line_1,gray,COLOR_BGR2GRAY);
                    adaptiveThreshold(gray, bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 105, 1); 
                    Mat dilates;
                    int morph_size = 2;
                    Mat element = getStructuringElement(
                        MORPH_RECT, Size(2 * morph_size + 1,
                                        2 * morph_size + 1),
                        Point(morph_size, morph_size));
                    Mat erod, dill;
                    dilate(bw,dilates, 0, Point(-1, -1), 2, 1, 1);
                    erode(bw, erod, element,Point(-1, -1), 1);
                    // resize(line_1,line_1,Size(94,94));
                    // imwrite("line_1.png",erod);
                    ocr->SetImage(erod.data,erod.cols,erod.rows,3, erod.step);
                    string result_1=string(ocr->GetUTF8Text());
                        cout<<"result_1 = "<<result_1<<std::endl;
                    //dong2
                    Mat crop_dong2_1=plate_cop(Rect(rect.x,rect.y+(int)(rect.height/2),rect.width,(int)(rect.height/2)));
                    Mat line_2=detect_text(crop_dong2_1);
                    // imwrite("line_2.png",line_2);
                    ocr->SetImage(line_2.data,line_2.cols,line_2.rows,3, line_2.step);
                    string results_2= string(ocr->GetUTF8Text());
                    cout<<"results_2 = "<<results_2<<std::endl;
                    string total= "OCR:"+result_1+results_2;
                    total.erase(std::remove(total.begin(), total.end(), '\n'), total.end());
                    cout<<""<<total<<std::endl;
                    }
            }
            else if (rect.width/rect.height >=2.5&&rect.width>0 && rect.height>0 &&rect.width+rect.x<=frame.cols&&rect.height+rect.y<=frame.rows && rect.x>0 &&rect.y>0)
            {
                cout<<"liscenPlate 1 line"<<std::endl;
                Mat crop_1line=plate_cop(Rect(rect.x,rect.y,rect.width,rect.height));
                Mat line_1dong=detect_text(crop_1line);
                // imwrite("line_1_dong.png",line_1dong);
                ocr->SetImage(line_1dong.data,line_1dong.cols,line_1dong.rows,3, line_1dong.step);
                string total=string(ocr->GetUTF8Text());
                total.erase(std::remove(total.begin(), total.end(), '\n'), total.end());
                cout<<""<<total<<std::endl;
            }
            }
            // Mat test=plate_cop(Rect(rect.x,rect.y,rect.width,rect.height));
            // Mat crop_plate = plate_cop(Rect(rect.x+60,rect.y+10,rect.width-100,(int)(rect.height/2)));
            // detect_text(test);
            // ocr->SetImage(crop_plate.data, crop_plate.cols, crop_plate.rows, 3, crop_plate.step);
            // string outText = string(ocr->GetUTF8Text());
            // cout<<"\n Results:"<<outText<<endl;
            // Mat crop_dong2=plate_cop(Rect(rect.x+25,rect.y+(int)(rect.height/2),rect.width-25,(int)(rect.height/2)));
            
            // ocr->SetImage(crop_dong2.data, crop_dong2.cols, crop_dong2.rows, 3, crop_dong2.step);
            // imwrite("i2.jpg",crop_dong2);
            // string outText2 = string(ocr->GetUTF8Text());
            // cout<<"\n Results:"<<outText2<<endl;
            // string results="OCR:"+outText+outText2;
            // results.erase(std::remove(results.begin(), results.end(), '\n'), results.end());
            // auto label_bg_sz = cv::getTextSize(results, cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
            // cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 5), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
            // cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
            // cv::putText(frame,results, cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
        }
    }
    auto total_end = std::chrono::steady_clock::now();
    float inference_fps =1000.0/std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
    float total_fps = 1000.0/std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    std::ostringstream stats_ss;
    stats_ss << std::fixed << std::setprecision(2);
    stats_ss << "Inference FPS : " << inference_fps << ", Total FPS: " << total_fps;
    // std::cerr<<"\nInferencetime : " << std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count()<<" ms"<<endl;
    auto stats = stats_ss.str();
    int baseline;
    auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));
    cv::namedWindow("output");
    cv::imshow("output", frame);
    // waitKey();
    char c=(char)waitKey(30);
    if (c=='q')
    break;
    }
cap.release();
destroyAllWindows();
ocr->End();
return 0;
}
