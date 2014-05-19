#ifndef FACERECOGNITION_H
#define FACERECOGNITION_H


#include "facedetector.h"

using namespace std;
using namespace cv;

const double DESIRED_LEFT_EYE_X = 0.16;
const double DESIRED_RIGHT_EYE_X = (1 - 0.16);
const double DESIRED_EYE_Y = 0.14;
const int DESIRED_FACE_WIDTH = 70;
const int DESIRED_FACE_HEIGHT = 70;

const int EIGEN_FACES  = 0;
const int LBP_FACE    = 1;
const int FISHER_FACE = 2;

const string EIGEN_FACE = "FaceRecognizer.Eigenfaces";
const string LBPH_FACE  = "FaceRecognizer.LBPH";
const string FISHER_FACES = "FaceRecognizer.Fisherfaces";


const float UNKNOWN_PERSON_THRESHOLD = 1.5;
const float CHANGE_FOR_ADD = 0.4;

class faceRecognition
{
public:

    faceRecognition();

    void addFace(Mat & faceMat,int faceLabel);

    void trainFacesMat();

    int  predictFaceMat(Mat & mat);

    string getClassifiedName();

    void preProcess(Mat & faceMat,Mat & outMat);

    bool newFaceAdded();

    bool hasEnoughDataToTrain() { return (addedFaceNum > 20);}

    bool isTrained();

    float getSimilarity(const Mat &mat1, const Mat &mat2);

    void reconstruction(Mat & preProcessedMat,Mat & reconstructedFace );

    void saveModel();

    void loadModel();

private:
    vector<Mat> processedFaceMat;

    vector<int> faceLabels;

    vector<string> className;

    faceDetector * detector;

    Mat preMat;

    void faceHisEqualization(Mat & inputMat,Mat & outputMat);

    void faceWarpAffine(Mat & inputMat,Mat & outputMat);

    void addEllpticalMask(Mat &);

    bool faceAdded;

    bool hasTrained;

    int addedFaceNum;

    Ptr<FaceRecognizer>  recognizer;

};

#endif // FACERECOGNITION_H
