

#include <jni.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "../../../../opencv/native/jni/include/opencv2/core/types.hpp"
#include "../../../../opencv/native/jni/include/opencv2/core/base.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <iterator>
//#include "histogram.hpp"




#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <string>
#include <android/log.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>


#define LOG_TAG "FaceDetection/DetectionBasedTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define BLOCK_SIZE 24



#define CELL_SIZE   32
#define FRAME_SIZE 164
#define DEFAULT_LBP_R 2
#define DEFAULT_LBP_P 8

using namespace cv;
using namespace std;


vector<int> convertToBinary(int x) {
    vector<int> result(8, 0);

    int idx = 0;
    while(x != 0) {
        result[idx] = x % 2;
        ++idx;
        x /= 2;
    }

    reverse(result.begin(), result.end());
    return result;
}

int countTransitions(vector<int> x) {
    int result = 0;
    for(int i = 0; i < 8; ++i)
        result += (x[i] != x[(i+1) % 8]);
    return result;
}

Mat uniformPatternHistogram(const Mat& src, int numPatterns) {
    Mat hist;
    hist = Mat::zeros(1, (numPatterns+1), CV_32SC1);

    for (int i = 0; i < numPatterns; ++i) {
        if (countTransitions(convertToBinary(i)) > 2)
            hist.at<int>(0, i) = -1;
    }

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            int bin = src.at<uchar>(i, j);
            if (hist.at<int>(0, bin) == -1)
                hist.at<int>(0, numPatterns) += 1;
            else
                hist.at<int>(0, bin) += 1;
        }
    }
    return hist;
}

void uniformPatternSpatialHistogram(const Mat& src, Mat& hist, int numPatterns,
                                    int gridX, int gridY, int overlap) {

    int width = src.cols;
    int height = src.rows;
    vector<Mat> histograms;

    Size window = Size(static_cast<int>(floor(src.cols/gridX)),
                       static_cast<int>(floor(src.rows/gridY)));

    for (int x = 0; x <= (width - window.width); x+= (window.width - overlap)) {
        for (int y = 0; y <= (height - window.height); y+= (window.height - overlap)) {
            Mat cell = Mat(src, Rect(x, y, window.width, window.height));
            histograms.push_back(uniformPatternHistogram(cell, numPatterns));
        }
    }

    hist.create(1, histograms.size()*(numPatterns+1), CV_32SC1);
    for (int histIdx = 0; histIdx < histograms.size(); ++histIdx) {
        for (int valIdx = 0; valIdx < (numPatterns+1); ++valIdx) {
            int y = (histIdx * (numPatterns+1)) + valIdx;
            hist.at<int>(0, y) = histograms[histIdx].at<int>(valIdx);
        }
    }
}

vector<int> getFeatureVector(Mat spatial_hist) {
    vector<int> feature_vector;
    for(int j = 0; j < spatial_hist.cols; ++j) {
        if(spatial_hist.at<int>(0, j) != -1)
            feature_vector.push_back(spatial_hist.at<int>(0, j));
    }
    return feature_vector;
}

void LBP(const Mat& src, Mat& dst) {


    const int iRows = src.rows;
    const int iCols = src.cols;

    //  cv::Mat resultMat(img_output.size(),img_output.type());

    //变量图像，生成LBP特征

    for(int y = 1;y<iRows-1; y++)
    {
        for(int x = 1;x<iCols-1; x++)
        {
            //定义8邻域
            uchar cNeighbor[8] = {0};
            cNeighbor[0] = src.at<uchar>(y-1,x-1);
            cNeighbor[1] = src.at<uchar>(y-1,x);
            cNeighbor[2] = src.at<uchar>(y-1,x+1);
            cNeighbor[3] = src.at<uchar>(y,  x+1);
            cNeighbor[4] = src.at<uchar>(y+1,x+1);
            cNeighbor[5] = src.at<uchar>(y+1,x);
            cNeighbor[6] = src.at<uchar>(y+1,x-1);
            cNeighbor[7] = src.at<uchar>(y  ,x-1);
            //当前图像的中心像素点
            uchar cCenter = src.at<uchar>(y,x);
            uchar cTemp   = 0;
            //计算LBP的值
            for(int k =0;k<8;k++)
            {
                cTemp += (cNeighbor[k]>=cCenter)*(1<<k);           //将1的二进制数按位左移k位
            }
            dst.at<uchar>(y,x) = cTemp;
        }//for x
    }
}



extern "C"

JNIEXPORT  jstring JNICALL
Java_com_example_myshite_MainActivity_imageprocessing(JNIEnv *env,
                                                      jobject instance,
                                                      jlong inputImage,
                                                      jlong outputImage,

                                                      jint th1,

                                                      jint th2 ,
                                                      jstring path) {
    // TODO: implement imageprocessing()
    vector<vector<Point> > contours;
    RNG rng(12345);
    vector<Vec4i> hierarchy;
    Mat &img_input = *(Mat *) inputImage;

    Mat &img_output = *(Mat *) outputImage;

    Mat &dst = *(Mat *) outputImage;

    cvtColor( img_input, dst, COLOR_RGB2GRAY);


    blur( dst, dst, Size(5,5) );
    /*
    vector<KeyPoint> v;
    Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(dst, v);
    for( unsigned int i = 0; i < v.size(); i++ )
    {
        const KeyPoint& kp = v[i];
        circle(img_input, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
    }*/
    jstring result;

    //double otsu_thresh_val = cv::threshold(img_input,img_input, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    // th1=otsu_thresh_val;
    //th2 = otsu_thresh_val * 0.5;

    // bilateralFilter(img_output, img_output,-1,sigmaColor,sigmaSpace);
    Canny( dst, dst, th1, th2);
//RETR_EXTERNAL
    findContours(dst, contours, hierarchy,CV_FILLED , CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    /*
    cv::Moments mom = cv::moments(contours[0]);
    double hu[7];
    cv::HuMoments(mom, hu); // now in hu are your 7 Hu-Moments
    Mat drawing = Mat::zeros(img_output.size(), CV_8UC3);
    vector<Point2f> mc(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, Point());
        circle(dst, mc[i], 4, color, -1, 8, 0);
    }*/
    /// Get the moments
    vector<Moments> mu(contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    { mu[i] = moments( contours[i], false ); }

    ///  Get the mass centers:
    vector<Point2f> mc( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

    /// Draw contours
   // Mat drawing = Mat::zeros( dst.size(), CV_8UC3 );
    //for( int i = 0; i< contours.size(); i++ )
  //  {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( dst, contours, -1, color, 2, 8, hierarchy, 0, Point() );
    //    circle( drawing, mc[i], 4, color, -1, 8, 0 );
    //}

/*
    /// Get the moments
    vector<Moments> mu(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        mu[i] = moments(contours[i], false);
    }

    ///  Get the mass centers:
    vector<Point2f> mc(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
    }

    /// Draw contours
    Mat drawing = Mat::zeros(img_output.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, Point());
        circle(dst, mc[i], 4, color, -1, 8, 0);
    }

    const int iRows = dst.rows;
    const int iCols = dst.cols;

    //  cv::Mat resultMat(img_output.size(),img_output.type());

    //变量图像，生成LBP特征

    for(int y = 1;y<iRows-1; y++)
    {
        for(int x = 1;x<iCols-1; x++)
        {
            //定义8邻域
            uchar cNeighbor[8] = {0};
            cNeighbor[0] = img_output.at<uchar>(y-1,x-1);
            cNeighbor[1] = img_output.at<uchar>(y-1,x);
            cNeighbor[2] = img_output.at<uchar>(y-1,x+1);
            cNeighbor[3] = img_output.at<uchar>(y,  x+1);
            cNeighbor[4] = img_output.at<uchar>(y+1,x+1);
            cNeighbor[5] = img_output.at<uchar>(y+1,x);
            cNeighbor[6] = img_output.at<uchar>(y+1,x-1);
            cNeighbor[7] = img_output.at<uchar>(y  ,x-1);
            //当前图像的中心像素点
            uchar cCenter = img_output.at<uchar>(y,x);
            uchar cTemp   = 0;
            //计算LBP的值
            for(int k =0;k<8;k++)
            {
                cTemp += (cNeighbor[k]>=cCenter)*(1<<k);           //将1的二进制数按位左移k位
            }
            img_output.at<uchar>(y,x) = cTemp;
        }//for x
    }*/
    //String &path1=*( String*)path; frommm herree bicha
     LBP(dst, img_output);
    Mat spatial_histogram;
      uniformPatternSpatialHistogram(img_output, spatial_histogram, 256, 3, 3, 0);

      vector<int> feature_vector = getFeatureVector(spatial_histogram);

      LOGD("r hist");

  /*
      FileStorage f;
      // code to write a cv::Mat to file
      f.open("hello.txt", FileStorage::WRITE);
      f << "m" <<  _histograms;
          f.release();


      FILE* file = fopen("/sdcard/hello.txt","w+");

      if (file != NULL)
      {
          fputs("hello", file);
          fflush(file);
          fclose(file);
      }
          __android_log_print(ANDROID_LOG_DEBUG,"path","%s",(path1 + "/info.txt").c_str());
          ofstream file;
          file.open((path1 + "/info.txt").c_str());
          for (int i = 0; i < _histograms.size(); i ++)
          {
              file.write(reinterpret_cast<char*>(&_histograms[i]), sizeof(float) * 100 * 1024);
          }
          file.close();
      FILE* file = NULL;
      file.open();
          for (int i = 0; i < _histograms.size(); i ++)
          {
              file.write(reinterpret_cast<char*>(&_histograms[i]), sizeof(float) * 100 * 1024);
          }

       file.close(); """""""""heeree comback"""""""""""""""
*/
    const char* file_path = (*env).GetStringUTFChars(path,NULL);
    if(file_path != NULL){
        LOGD("From c file_path %s", file_path);
    }

    //打开文件
    FILE* file = fopen(file_path, "w+");
    if(file != NULL){
        LOGD("From c open file success");
    }

    //写入文件
    char data[BLOCK_SIZE]= "I am a child";

    int count = fwrite(data, BLOCK_SIZE , 1, file);
    if(count > 0){
        LOGD("Frome c write file success");
    }

    //关闭文件
    if(file != NULL){
        fclose(file);
    }

    (*env).ReleaseStringUTFChars(path, file_path);



    stringstream ss;
    copy( feature_vector.begin(), feature_vector.end(), ostream_iterator<int>(ss, " "));
    string s1 = ss.str();
    s1 = s1.substr(0, s1.length()-1);

    char cstr[s1.size() + 1];

    std::copy(s1.begin(), s1.end(), cstr);
    cstr[s1.size()] = '\0';
    LOGD("finish");

    return env->NewStringUTF("finish");

}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_myshite_App_StartActivity_LbpCal(JNIEnv *env, jclass StartActivity, jlong input_image,
                                                  jlong output_image ,jstring path) {
    // TODO: implement LbpCal()
    vector<vector<Point> > contours;
    RNG rng(12345);
    vector<Vec4i> hierarchy;
    Mat &img_input = *(Mat *) input_image;

    Mat &img_output = *(Mat *) output_image;

    Mat &dst = *(Mat *) output_image;

    cvtColor( img_input, dst, COLOR_RGB2GRAY);


    GaussianBlur( dst, dst, Size(5,5),0,0 );
    Canny( dst, dst, THRESH_OTSU, THRESH_OTSU*2);

    findContours(dst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    cv::Moments mom = cv::moments(contours[0]);
    double hu[7];
    cv::HuMoments(mom, hu); // now in hu are your 7 Hu-Moments
    Mat drawing = Mat::zeros(img_output.size(), CV_8UC3);
    vector<Point2f> mc(contours.size());
    vector<Moments> mu(contours.size());

    int largest_area=0;
    int largest_contour_index=-1;

    for( size_t i = 0; i< contours.size(); i++ ) // iterate through each contour.
    {
        int area = contourArea( contours[i] );  //  Find the area of contour

        if( area > largest_area )
        {
            largest_area = area;
            largest_contour_index = i;
        }
    }

    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    // drawContours(dst, contours, largest_contour_index, color);
    drawContours( dst, contours,largest_contour_index, Scalar( 0, 255, 0 ), 2 );
    LBP(dst, img_output);
    Mat spatial_histogram;
    uniformPatternSpatialHistogram(img_output, spatial_histogram, 256, 3, 3, 0);

    vector<int> feature_vector = getFeatureVector(spatial_histogram);

    LOGD("run hist");
    const char* file_path = (*env).GetStringUTFChars(path,NULL);
    if(file_path != NULL){
        LOGD("From c file_path %s", file_path);
    }

    //打开文件
    FILE* file = fopen(file_path, "w+");
    if(file != NULL){
        LOGD("From c open file success");
    }

    //写入文件
    char data[BLOCK_SIZE]= "I am a child";

    int count = fwrite(data, BLOCK_SIZE , 1, file);
    if(count > 0){
        LOGD("Frome c write file success");
    }

    //关闭文件
    if(file != NULL){
        fclose(file);
    }

    (*env).ReleaseStringUTFChars(path, file_path);



    stringstream ss;
    copy( feature_vector.begin(), feature_vector.end(), ostream_iterator<int>(ss, " "));
    string s1 = ss.str();
    s1 = s1.substr(0, s1.length()-1);

    char cstr[s1.size() + 1];

    std::copy(s1.begin(), s1.end(), cstr);
    cstr[s1.size()] = '\0';
    LOGD("finish");

    return env->NewStringUTF("finish");
}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_App_StartActivity_Orbcal(JNIEnv *env, jobject thiz, jlong input_image,
                                                  jlong output_image) {
    // TODO: implement Orbcal()
    Mat &captured = *(Mat *)  input_image;

    Mat &target = *(Mat *) output_image;
    // Mat &target = *(Mat *) img2;
    BFMatcher matcher(NORM_L2);

    Ptr<ORB> orb = ORB::create();
    std::vector<cv::KeyPoint> keypointsCaptured;
    //std::vector<cv::KeyPoint> keypointsTarget;

    cv::Mat descriptorsCaptured;

    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> symMatches;
    orb = ORB::create();

//Pre-process
    //Mat &MatchesImage = *(Mat *) image3;

   // medianBlur(captured, target, 5);
    Mat &dst = *(Mat *) output_image;

    cvtColor( captured, dst, COLOR_RGB2GRAY);

    vector<vector<Point> > contours;
    RNG rng(12345);
    vector<Vec4i> hierarchy;
    GaussianBlur( dst, dst, Size(5,5),0,0 );
    Canny( dst, dst, THRESH_OTSU, THRESH_OTSU*2);

    findContours(dst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    cv::Moments mom = cv::moments(contours[0]);
    double hu[7];
    cv::HuMoments(mom, hu); // now in hu are your 7 Hu-Moments
    //Mat drawing = Mat::zeros(img_output.size(), CV_8UC3);
    vector<Point2f> mc(contours.size());
    vector<Moments> mu(contours.size());

    int largest_area=0;
    int largest_contour_index=-1;

    for( size_t i = 0; i< contours.size(); i++ ) // iterate through each contour.
    {
        int area = contourArea( contours[i] );  //  Find the area of contour

        if( area > largest_area )
        {
            largest_area = area;
            largest_contour_index = i;
        }
    }

    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    // drawContours(dst, contours, largest_contour_index, color);
    drawContours( dst, contours,largest_contour_index, Scalar( 0, 255, 0 ), 2 );
    orb->detectAndCompute(dst, noArray(), keypointsCaptured, descriptorsCaptured);
    if(descriptorsCaptured.type()!=CV_32F) {
        descriptorsCaptured.convertTo(descriptorsCaptured, CV_32F); }
     drawKeypoints(captured, keypointsCaptured, target, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    FileStorage fs;
    fs.open("/sdcard/descriptors.xml", FileStorage::WRITE);
    fs << "Des" << descriptorsCaptured;
    fs.release();
    /*
    cv::Size size = descriptorsCaptured.size();

    int total = size.width * size.height* descriptorsCaptured.channels();


    std::vector<uchar> data(descriptorsCaptured.ptr(), descriptorsCaptured.ptr() + total);
    string s(data.begin(), data.end());
    LOGI("Done writing descriptors.\n");

return s;*/
    __android_log_print(ANDROID_LOG_INFO, "sometag", "keypoints size = %d", keypointsCaptured.size());

}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_App_StartActivity_DetectEdge(JNIEnv *env, jobject thiz, jlong input_image,
                                                      jlong output_image) {
    // TODO: implement DetectEdge()

    vector<vector<Point> > contours;
    RNG rng(12345);
    vector<Vec4i> hierarchy;
    Mat &img_input = *(Mat *) input_image;

    Mat &img_output = *(Mat *) output_image;

    Mat &dst = *(Mat *) output_image;

    cvtColor( img_input, dst, COLOR_RGB2GRAY);


    GaussianBlur( dst, dst, Size(5,5),0,0 );
    //dilate()
    /* minee*********************************************************
    vector<KeyPoint> v;
    Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(dst, v);
    for( unsigned int i = 0; i < v.size(); i++ )
    {
        const KeyPoint& kp = v[i];
        circle(img_input, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
    ***************************************************}*/


    //double otsu_thresh_val = cv::threshold(img_input,img_input, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    // th1=otsu_thresh_val;
    //th2 = otsu_thresh_val * 0.5;

    // bilateralFilter(img_output, img_output,-1,sigmaColor,sigmaSpace);
    Canny( dst, dst, THRESH_OTSU, THRESH_OTSU*2);
   // CV_FILLED
    findContours(dst, contours, hierarchy,  RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    ///  Get the mass centers:


    /*
    cv::Moments mom = cv::moments(contours[0]);
    double hu[7];
    cv::HuMoments(mom, hu); // now in hu are your 7 Hu-Moments
    Mat drawing = Mat::zeros(img_output.size(), CV_8UC3);
    vector<Point2f> mc(contours.size());
    vector<Moments> mu(contours.size());



    /*for (int i = 0; i < contours.size(); i++)
    {
        mu[i] = moments(contours[i], false);
    }

    ///  Get the mass centers:


    for (int i = 0; i < contours.size(); i++)
    {
        mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
    }

 */

     int largest_area=0;
    int largest_contour_index=-1;

    for( size_t i = 0; i< contours.size(); i++ ) // iterate through each contour.
    {
        int area = contourArea( contours[i] );  //  Find the area of contour

        if( area > largest_area )
        {
            largest_area = area;
            largest_contour_index = i;               //Store the index of largest contour
            //bounding_rect = boundingRect( contours[i] ); // Find the bounding rectangle for biggest contour
        }
    }

    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    // drawContours(dst, contours, largest_contour_index, color);
    drawContours( dst, contours, largest_contour_index, Scalar( 0, 255, 0 ), 2 );
    // circle(dst, mc[largest_contour_index], 4, color, -1, 8, 0);


}