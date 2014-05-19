#include "facerecognition.h"
#include <string>

faceRecognition::faceRecognition():faceAdded(false),hasTrained(false),addedFaceNum(0),detector(new faceDetector)
{
    recognizer = Algorithm::create<FaceRecognizer>(EIGEN_FACE);
    loadModel();
}

void faceRecognition::preProcess(Mat & faceMat,Mat & outMat)
{
    Mat tmpFaceMat;
    if(faceMat.channels() == 3)
    {
        cvtColor(faceMat,tmpFaceMat,CV_BGR2GRAY);
    }
    else if(faceMat.channels() == 4)
    {
        cvtColor(faceMat,tmpFaceMat,CV_BGRA2GRAY);
    }
    else
    {
        tmpFaceMat = faceMat;
    }

    faceWarpAffine(tmpFaceMat,outMat);
    faceHisEqualization(outMat,outMat);
    addEllpticalMask(outMat);
}

void faceRecognition::faceWarpAffine(Mat & inputMat,Mat & outputMat)
{
    Mat tmpFaceMat ;
    inputMat.copyTo(tmpFaceMat);
    vector<Rect> eyeRect;
    Rect leftEyeRect(-1,-1,-1,-1);
    Rect rightEyeRect(-1,-1,-1,-1);

    Point2f eyeCenter;
    Point2f leftEye(-1,-1);
    Point2f rightEye(-1,-1);

    if(detector->detectLeftEye(tmpFaceMat,eyeRect))
    {
        leftEyeRect = eyeRect[0];
    }

    if(detector->detectRightEye(tmpFaceMat,eyeRect))
    {
        rightEyeRect = eyeRect[0];
        rightEyeRect.x += tmpFaceMat.cols/2;
    }

    leftEye.x = leftEyeRect.x + leftEyeRect.width*0.5;
    leftEye.y = leftEyeRect.y + leftEyeRect.height*0.5;
    rightEye.x = rightEyeRect.x + rightEyeRect.width*0.5;
    rightEye.y = rightEyeRect.y + rightEyeRect.height*0.5;

    eyeCenter.x = (leftEye.x + rightEye.x)*0.5;
    eyeCenter.y = (leftEye.y + rightEye.y)*0.5;

    double dx = (rightEye.x - leftEye.x);
    double dy = (rightEye.y - leftEye.y);
    double len = sqrt(dx*dx + dy*dy);
    double angle = atan2(dy,dx)*180/CV_PI;

    double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X)*DESIRED_FACE_WIDTH;

    double scale = desiredLen/len;

    Mat rot_Mat = getRotationMatrix2D(eyeCenter,angle,scale);

    double ex = DESIRED_FACE_WIDTH*0.5 - eyeCenter.x;
    double ey = DESIRED_FACE_HEIGHT*DESIRED_EYE_Y - eyeCenter.y;

    rot_Mat.at<double>(0,2) += ex;
    rot_Mat.at<double>(1,2) += ey;

    outputMat = Mat(DESIRED_FACE_WIDTH,DESIRED_FACE_HEIGHT,CV_8U,Scalar(128));
    warpAffine(tmpFaceMat,outputMat,rot_Mat,outputMat.size());
}

 void faceRecognition::faceHisEqualization(Mat & inputMat,Mat & outputMat)
 {
    int W = inputMat.cols;
    int H = inputMat.rows;
    Mat wholeFace;
    equalizeHist(inputMat,wholeFace);

    int midW = W/2;
    Mat leftFace = inputMat(Rect(0,0,midW,H));
    Mat rightFace = inputMat(Rect(midW,0,W - midW,H));

    equalizeHist(leftFace,leftFace);
    equalizeHist(rightFace,rightFace);

    Mat HisEquFace;
    HisEquFace.create(inputMat.size(),inputMat.type());

    for(int y = 0 ;y < H; ++y)
    {
        for(int x = 0 ; x < W; ++x)
        {
            int v;
            if(x < W/4)
            {
                v = leftFace.at<uchar>(y,x);
            }
            else if(x < W*2/4)
            {
                int lv = leftFace.at<uchar>(y,x);
                int wv = wholeFace.at<uchar>(y,x);

                float f = (x - W*1/4)/(float)(W/4);
                v = cvRound((1.0 - f)*lv + (f)*wv);
            }
            else if(x < W*3/4)
            {
                int rv = rightFace.at<uchar>(y,x - midW);
                int wv = wholeFace.at<uchar>(y,x);

                float f = (x - W*2/4)/(float)(W/4);
                v = cvRound((1.0 - f)*wv + (f)*rv);
            }
            else
            {
                v = rightFace.at<uchar>(y,x - midW);
            }
            HisEquFace.at<uchar>(y,x) = v;

        }
    }

    outputMat.create(HisEquFace.size(),HisEquFace.type());
    bilateralFilter(HisEquFace,outputMat,0,20.0,2.0);
 }

 void faceRecognition::addEllpticalMask(Mat & faceMat)
 {
     Mat mask = Mat(faceMat.size(),CV_8UC1,Scalar(255));
     int dw = DESIRED_FACE_WIDTH;
     int dh = DESIRED_FACE_HEIGHT;
     Point faceCenter = Point(cvRound(dw*0.5),cvRound(dh*0.4));

     Size size = Size(cvRound(dw*0.5),cvRound(dh*0.8));
     ellipse(mask,faceCenter,size,0,0,360,Scalar(0),CV_FILLED);
     faceMat.setTo(Scalar(128),mask);
 }


void faceRecognition::addFace(Mat & faceMat,int faceLabel)
{
    Mat preProcessedMat;
    Mat flipFace;
    preProcess(faceMat,preProcessedMat);
    if(preMat.data)
    {
        float similarity = getSimilarity(preProcessedMat,preMat);
        if(similarity < CHANGE_FOR_ADD)
        {
            return ;
        }
        cv::swap(preMat,preProcessedMat);
    }
    flip(preProcessedMat,flipFace,0);
    processedFaceMat.push_back(preProcessedMat);
    processedFaceMat.push_back(flipFace);
    faceLabels.push_back(faceLabel);
    faceLabels.push_back(faceLabel);
    addedFaceNum ++;
    faceAdded = true;
}

bool faceRecognition::newFaceAdded()
{
    return faceAdded;
}

void faceRecognition::trainFacesMat()
{
    recognizer->train(processedFaceMat,faceLabels);
    addedFaceNum = 0;
    hasTrained = true;
    faceAdded = false;
}

int  faceRecognition::predictFaceMat(Mat & mat)
{
    Mat outMat;
    Mat reconstructedFace;
    preProcess(mat,outMat);

    double confidence;
    int label;
    recognizer->predict(outMat,label,confidence);
    reconstruction(outMat,reconstructedFace);

    float similarity = getSimilarity(outMat,reconstructedFace);
    cout<<"similarity:"<<similarity<<endl;
    if(similarity > UNKNOWN_PERSON_THRESHOLD)
    {
        label = -1;
    }

    return label;
}

float faceRecognition::getSimilarity(const Mat &mat1, const Mat &mat2)
{
    if(mat1.cols > 0 && mat1.rows > 0&& (mat1.cols == mat2.cols)&&(mat1.rows == mat2.rows))
    {
        double errorL2 = norm(mat1,mat2,CV_L2);
        double similarity = errorL2/(double)(mat1.cols*mat2.rows);
        return similarity;
    }
    else
    {
        return 100000.0;
    }
}

void faceRecognition::reconstruction(Mat & preProcessedMat,Mat & reconstructedFace )
{
    Mat eigenFace = recognizer->get<Mat>("eigenvectors");
    Mat averageFace = recognizer->get<Mat>("mean");

    Mat projection = subspaceProject(eigenFace,averageFace,preProcessedMat.reshape(1,1));
    Mat reconstructionRow = subspaceReconstruct(eigenFace,averageFace,projection);

    Mat reconstructedMat = reconstructionRow.reshape(1,preProcessedMat.rows);
    reconstructedFace = Mat(reconstructedMat.size(),CV_8U);
    reconstructedMat.convertTo(reconstructedFace,CV_8U,1,0);
}

bool faceRecognition::isTrained()
{
    return hasTrained;
}

void faceRecognition::saveModel()
{
    recognizer->save("trainedModel.yml");
}

void faceRecognition::loadModel()
{
    try
    {
        recognizer->load("trainedModel.yml");
    }
    catch(cv::Exception &e)
    {
        cout<<"no trained Model"<<endl;
        return ;
    }

    hasTrained = true;
}

