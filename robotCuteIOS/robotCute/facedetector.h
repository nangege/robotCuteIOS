#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H


#include<iostream>
#include <algorithm>

using namespace std;
using namespace cv;

//const string  FACE_CASCADE  = "../haarcascade_frontalface_alt.xml";
const string predict = "/";

const string FACE_CASCADE = predict + "haarcascade_frontalface_alt2.xml";
const string EYE_TREE_CASCADE = predict +"haarcascade_eye_tree_eyeglasses.xml";
const string  EYE_CASCADE = predict +"haarcascade_eye.xml";
const string  NOSE_CASCADE =  predict + "haarcascade_mcs_nose.xml";
const string  MOUSE_CASCADE = predict +"haarcascade_mcs_mouth.xml";
const string  LEFT_EAR_CASCADE  = predict +"haarcascade_mcs_leftear.xml";
const string  RIGHT_EAR_CASCADE = predict + "haarcascade_mcs_rightear.xml";

const int DETECT_FACE = 0;
const int DETECT_EYE = 1;
const int DETECT_EYE_GLASS = 2;
const int DETECT_NOSE      = 3;
const int DETECT_MOUSE     = 4;
const int DETECT_LEFT_EAR  = 5;
const int DETECT_RIGHT_EAR = 6;

const int WIDTH =  90;
const int HEIGHT = 120;



#define DEBUG_DETECT

class faceDetector
{
public:
    faceDetector();

    void setMatToDetect(const Mat & image);

    void detectFace();
    void detectLeftEye();
    bool detectLeftEye(Mat & faceMat,vector<cv::Rect> &leftEyes);
    void detectRightEye();
    bool detectRightEye(Mat & faceMat,vector<cv::Rect> &rightEyes);
    void detectNose();
    void detectMouth();
    void detectAllFeatures();

    bool isFaceDetected(){return FaceDetected;}

    cv::Rect getFaceRect();
    Mat  getFaceMat();

    vector<cv::Rect> getAllFace();

    static bool rectSort(const cv::Rect & rect1,const cv::Rect & rect2);

    cv::Rect getNoseRect();
    cv::Rect getRelateiveNoseRect();

    cv::Rect getMouthRect();
    cv::Rect getRelativeMouthRect();

    cv::Rect getLeftEyeRect();
    cv::Rect getRelativeLEyeRect();

    cv::Rect getRightEyeRect();
    cv::Rect getRelativeREyeRect();

    const Mat & getOriginalMat(){return originalMat;}
    const Mat & getOriginalFaceMat();


    void initClassifier();
    void preProcess();

    void detectObject(CascadeClassifier & classifier,const Mat &input,vector<cv::Rect>  &objects);
    void detectLargestObject(CascadeClassifier & classifier,const Mat & input ,vector<cv::Rect> & object);
    
    void setObjSize(float size){objSize = size;}
    void setpreString(string  str){ preString = str;};

private:
    CascadeClassifier faceClassifier;
    CascadeClassifier eyesClassifier;
    CascadeClassifier glassEyesClassifier;
    CascadeClassifier noseClassifier;
    CascadeClassifier mouthCalssifier;

    Mat grayImage;
    Mat originalMat;
    Mat originalFaceMat;

    bool FaceEnable;
    bool EyeEnable;
    bool GlassEyeEnable;
    bool NoseEnable;
    bool MouthEnable;

    bool FaceDetected;

    float widthScale;
    float heightScale;
    
    float objSize ;

    vector<cv::Rect> faces;
    vector<cv::Rect> leftEyes;
    vector<cv::Rect> rightEyes;
    vector<cv::Rect> noses;
    vector<cv::Rect> mouth;

    cv::Rect FaceRIO;
    cv::Rect leftEye;
    cv::Rect rightEye;
    cv::Rect noseRect;
    cv::Rect mouthRect;

    Mat faceMat ;
    
    string preString;
};

#endif // FACEDETECTOR_H
