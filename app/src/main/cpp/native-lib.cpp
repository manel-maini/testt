
/*
#include <jni.h>

#include "com_example_android_opencvdemo_OpenCvMaker.h"


extern "C" JNIEXPORT jint JNICALL Java_com_example_myshite_OpenCvMaker_makeGray
        (JNIEnv *, jclass, jlong addrInput, jlong addrOutput){
    return (jint) toGray((*(Mat*) addrInput),(*(Mat*) addrOutput));
}

int toGray(Mat img, Mat& gray) {
    cv::cvtColor(img, gray,CV_RGBA2GRAY);
    if(gray.rows == img.rows && gray.cols == img.cols)
        return 1;
    return 0;
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_myshite_OpenCvMaker_makeCanny
        (JNIEnv *, jclass, jlong addrInput, jlong addrOutput){
    return (jint) toCanny((*(Mat*) addrInput),(*(Mat*) addrOutput));
}

int toCanny(Mat input, Mat& output) {
    Canny(input,output,75,150,3,false);
    if(output.rows == input.rows && output.cols == input.cols)
        return 1;
    return 0;
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_myshite_OpenCvMaker_makeDilate
        (JNIEnv *, jclass, jlong addrInput, jlong addrOutput){
    return (jint) toDilate((*(Mat*) addrInput),(*(Mat*) addrOutput));
}

int toDilate(Mat input, Mat& output){
    dilate(input,output,Mat(), Point(-1,-1),2,1,1);
    if(output.rows == input.rows && output.cols == input.cols)
        return 1;
    return 0;
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_myshite_OpenCvMaker_makeErode
        (JNIEnv *, jclass, jlong addrInput, jlong addrOutput){
    return (jint) toErode((*(Mat*) addrInput),(*(Mat*) addrOutput));
}

int toErode(Mat input, Mat& output){
    erode(input,output,Mat(), Point(-1,-1),2,1,1);
    if(output.rows == input.rows && output.cols == input.cols)
        return 1;
    return 0;
}
*/
#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <sstream>
#include <android/log.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

//#include "JHashMap.h"
#define LOG_TAG "FaceDetection/DetectionBasedTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
using namespace std;
using namespace cv;
using namespace cv::ml;

#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))


extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_Kmeans_kmeans(JNIEnv *env, jclass clazz, jlong img1) {



    Mat &captured = *(Mat *)  img1;

   // Mat &target = *(Mat *) img2;
    BFMatcher matcher(NORM_L2);

    Ptr<ORB> orb = ORB::create();
    std::vector<cv::KeyPoint> keypointsCaptured;
    std::vector<cv::KeyPoint> keypointsTarget;

    cv::Mat descriptorsCaptured;
    cv::Mat descriptorsTarget;
//cv::Mat captured;
    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> symMatches;
    orb = ORB::create();

//Pre-process
    //Mat &MatchesImage = *(Mat *) image3;

    medianBlur(captured, captured, 5);
/*
    cv::Mat reshaped_image = captured.reshape(1, captured.cols * captured.rows);
    cv::Mat reshaped_image32f;
    reshaped_image.convertTo(reshaped_image32f, CV_32FC1, 1.0 / 255.0);
//////////if(descriptors_1.type()!=CV_32F) {
//////////////////descriptors_1.convertTo(descriptors_1, CV_32F); }

 /*
 Mat labels = new Mat(contours.size(), 2, CvType.CV_32SC1);
    TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 100, 1.0);
    Mat centers = new Mat();
    Core.kmeans(samples32final, 5, labels, criteria, 10, Core.KMEANS_PP_CENTERS, centers);
    cv::Mat labels;
    int cluster_number = 5;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT, 100, 1);
    cv::Mat centers;
    cv::kmeans(reshaped_image32f, cluster_number, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);
*/
    // medianBlur(target, target, 5);

    orb->detectAndCompute(captured, noArray(), keypointsCaptured, descriptorsCaptured);
    if(descriptorsCaptured.type()!=CV_32F) {
        descriptorsCaptured.convertTo(descriptorsCaptured, CV_32F); }
   // drawKeypoints(captured, keypointsCaptured, target, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    FileStorage fs;
    fs.open("/sdcard/descriptors.xml", FileStorage::WRITE);
    fs << "Des" << descriptorsCaptured;
    fs.release();


    cv::Mat labels;
    int cluster_number = 5;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT, 100, 1);
    cv::Mat centers;
    cv::kmeans(descriptorsCaptured, cluster_number, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);


    LOGI("Done writing descriptors.\n");
    // orb->detectAndCompute(target, noArray(), keypointsTarget, descriptorsTarget);
    __android_log_print(ANDROID_LOG_INFO, "sometag", "keypoints2 size = %d", keypointsTarget.size());
    __android_log_print(ANDROID_LOG_INFO, "sometag", "keypoints size = %d", keypointsCaptured.size());

//Match images based on k nearest neighbour
    // std::vector<std::vector<cv::DMatch> > matches1;
    //   matcher.knnMatch(descriptorsCaptured , descriptorsTarget,
    // matches1, 2);
//__android_log_print(ANDROID_LOG_INFO, "sometag", "Matches1 = %d",     matches1.size());
    std::vector<std::vector<cv::DMatch> > matches2;
    matcher.knnMatch(descriptorsTarget , descriptorsCaptured,
                     matches2, 2);
    //Mat &MatchesImage = *(Mat *) image3;
    LOGD("run hist");

    //  drawMatches(captured,keypointsCaptured, target, keypointsTarget, matches, MatchesImage, cv::Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    /* Mat_<float> trainingData(10, 2);
     trainingData <<  0, 0,
             0.5, 0,
             0.5, 0.25,
             1, 1.25,
             1, 1.5,
             1, 1,
             0.5, 1.5,
             0.25, 1,
             2, 1.5,
             2, 2.5;

     Mat_<int> trainingLabels(1, 10);
     trainingLabels << 0, 0, 0, 1, 1, 1, 0, 1, 1, 1;

     Ptr<SVM> svm = SVM::create();
     svm->setType(SVM::C_SVC);
     svm->setKernel(SVM::LINEAR);
     svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

     svm->train(trainingData, ROW_SAMPLE, trainingLabels);

     Mat_<float> testFeatures(1, 2);
     testFeatures << 2.5, 2.5;

     Mat res;
     svm->predict(testFeatures, res);*/



}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_orbMatchTry_compareImages(JNIEnv *env, jclass clazz, jlong img1,
                                                   jlong img2, jlong image3) {



    Mat &captured = *(Mat *)  img1;

    BFMatcher matcher(NORM_L2);
    Mat &target = *(Mat *) img2;
    Ptr<ORB> orb = ORB::create();
    std::vector<cv::KeyPoint> keypointsCaptured;
    std::vector<cv::KeyPoint> keypointsTarget;

    cv::Mat descriptorsCaptured;
    cv::Mat descriptorsTarget;
//cv::Mat captured;
    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> symMatches;
    orb = ORB::create();

//Pre-process
    Mat &MatchesImage = *(Mat *) image3;

    medianBlur(captured, captured, 5);
/*
    cv::Mat reshaped_image = captured.reshape(1, captured.cols * captured.rows);
    cv::Mat reshaped_image32f;
    reshaped_image.convertTo(reshaped_image32f, CV_32FC1, 1.0 / 255.0);

    cv::Mat labels;
    int cluster_number = 5;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT, 100, 1);
    cv::Mat centers;
    cv::kmeans(reshaped_image32f, cluster_number, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);
*/
    medianBlur(target, target, 5);

    orb->detectAndCompute(captured, noArray(), keypointsCaptured, descriptorsCaptured);
    orb->detectAndCompute(target, noArray(), keypointsTarget, descriptorsTarget);
    __android_log_print(ANDROID_LOG_INFO, "sometag", "keypoints2 size = %d", keypointsTarget.size());
    __android_log_print(ANDROID_LOG_INFO, "sometag", "keypoints size = %d", keypointsCaptured.size());

//Match images based on k nearest neighbour
    std::vector<std::vector<cv::DMatch> > matches1;
    matcher.knnMatch(descriptorsCaptured , descriptorsTarget,
                     matches1, 2);

    //Mat &MatchesImage = *(Mat *) image3;
    LOGD("run hist");
    /*
    double dist_th = 64;
    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance < dist_th )
        { good_matches.push_back( matches[i]); }
    }*/
//cluster the feature vectors

    /*
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)captured.cols, 0 );
    obj_corners[2] = Point2f( (float)captured.cols, (float)captured.rows );
    obj_corners[3] = Point2f( 0, (float)captured.rows );
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H);
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( target, scene_corners[0] + Point2f((float)captured.cols, 0),
          scene_corners[1] + Point2f((float)captured.cols, 0), Scalar(0, 255, 0), 4 );
    line( target, scene_corners[1] + Point2f((float)captured.cols, 0),
          scene_corners[2] + Point2f((float)captured.cols, 0), Scalar( 0, 255, 0), 4 );
    line( target, scene_corners[2] + Point2f((float)captured.cols, 0),
          scene_corners[3] + Point2f((float)captured.cols, 0), Scalar( 0, 255, 0), 4 );
    line( target, scene_corners[3] + Point2f((float)captured.cols, 0),
          scene_corners[0] + Point2f((float)captured.cols, 0), Scalar( 0, 255, 0), 4 );
    std::vector< DMatch > good_matches;
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for (size_t i = 0; i < matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back(keypointsCaptured[matches[i].queryIdx].pt);
        scene.push_back(keypointsTarget[matches[i].trainIdx].pt);
    }
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0, 0);
    obj_corners[1] = Point(captured.cols, 0);
    obj_corners[2] = Point(captured.cols,captured.rows);
    obj_corners[3] = Point(0, captured.rows);
    std::vector<Point2f> scene_corners(4);

    Mat H = findHomography(obj, scene, RANSAC);
    perspectiveTransform(obj_corners, scene_corners, H);
    std::vector<Point2f> scene_corners_ ;
    scene_corners_ = scene_corners;

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line(MatchesImage,
         scene_corners[0] + Point2f((float)captured.cols, 0), scene_corners[1] + Point2f((float)captured.cols, 0),
         Scalar(0, 255, 0), 2, LINE_AA);
    line(MatchesImage,
         scene_corners[1] + Point2f((float)captured.cols, 0), scene_corners[2] + Point2f((float)captured.cols, 0),
         Scalar(0, 255, 0), 2, LINE_AA);
    line(MatchesImage,
         scene_corners[2] + Point2f((float)captured.cols, 0), scene_corners[3] + Point2f((float)captured.cols, 0),
         Scalar(0, 255, 0), 2, LINE_AA);
    line(MatchesImage,
         scene_corners[3] + Point2f((float)captured.cols, 0), scene_corners[0] + Point2f((float)captured.cols, 0),
         Scalar(0, 255, 0), 2, LINE_AA);*/

}
