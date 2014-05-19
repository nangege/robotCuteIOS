#include "facedetector.h"

faceDetector::faceDetector():FaceEnable(false),EyeEnable(false),NoseEnable(false),MouthEnable(false),FaceDetected(false),widthScale(1.0),heightScale(1.0),objSize(60.0),preString("")
{
   
}

void faceDetector::detectAllFeatures()
{
    detectFace();
    detectLeftEye();
    detectRightEye();
    detectNose();
    detectMouth();
}

void faceDetector::setMatToDetect(const Mat &image)
{
    image.copyTo(originalMat);

    if(originalMat.channels() == 3)
    {
        cvtColor(originalMat,grayImage,CV_RGB2GRAY);
    }
    else if(image.channels() == 4)
    {
        cvtColor(originalMat,grayImage,CV_RGBA2GRAY);
    }
    else
    {
        grayImage = image;
    }

    FaceDetected = false;
    FaceRIO = Rect(-1,-1,-1,-1);

    preProcess();

}

void faceDetector::initClassifier()
{
    if(faceClassifier.load(preString + FACE_CASCADE))
    {
        FaceEnable = true;
    }
    else
    {
        cout<<"face XML load Filed"<<endl;
    }

    if(eyesClassifier.load(preString + EYE_CASCADE))
    {
        EyeEnable = true;
    }
    else
    {
        cout<<"Eye XML load Filed"<<endl;
    }
    if(glassEyesClassifier.load(preString + EYE_TREE_CASCADE))
    {
        GlassEyeEnable = true;
    }
    else
    {
        cout<<"glassEye XML load Filed"<<endl;
    }
    if(noseClassifier.load(preString + NOSE_CASCADE))
    {
        NoseEnable = true;
    }
    else
    {
        cout<<"nose XML load Filed"<<endl;
    }

    if(mouthCalssifier.load(preString + MOUSE_CASCADE))
    {
        MouthEnable = true;
    }
    else
    {
        cout<<"mouth XML load Filed"<<endl;
    }
}

void faceDetector::preProcess()
{
    widthScale = heightScale = grayImage.cols/WIDTH;
    //heightScale = grayImage.rows/HEIGHT;
    int height = grayImage.rows/widthScale;
    
    resize(grayImage,grayImage,Size(WIDTH,height));
    //resize(originalMat,originalMat,Size(WIDTH,HEIGHT));

    //blur(grayImage,grayImage,Size(3,3));
    //equalizeHist(grayImage,grayImage);
}

bool faceDetector::rectSort(const Rect &rect1, const Rect &rect2)
{
    return rect1.width > rect2.height;
}
void faceDetector::detectFace()
{
    if(FaceEnable)
    {
        detectObject(faceClassifier,grayImage,faces);
    }

    leftEyes.clear();
    rightEyes.clear();
    noses.clear();
    mouth.clear();

    std::sort(faces.begin(),faces.end(),rectSort);

    if(faces.size() > 0)
    {
        FaceRIO = faces[0];
        FaceDetected = true;
        faceMat = grayImage(FaceRIO);
#ifdef DEBUG_DETECT
        cout<<"Detected!"<<endl;
#endif
    }
    else
    {
        FaceRIO = Rect(-1,-1,-1,-1);
#ifdef DEBUG_DETECT
        cout<<"Not Detected!"<<endl;
#endif
    }
}

vector<Rect> faceDetector::getAllFace()
{
    vector<Rect> allFace = faces;
    for(int i = 0; i < allFace.size(); ++i)
    {
        allFace[i].x = allFace[i].x*widthScale;
        allFace[i].y = allFace[i].y*heightScale;
        allFace[i].width = allFace[i].width *widthScale;
        allFace[i].height = allFace[i].height*heightScale;
    }
    return allFace;
}
Mat faceDetector::getFaceMat()
{
    if(isFaceDetected())
    {
        return grayImage(FaceRIO);
    }
    else
    {
        return Mat(16,16,CV_8UC1);
    }
}

const Mat & faceDetector::getOriginalFaceMat()
{
    if(isFaceDetected())
    {
        originalFaceMat = originalMat(FaceRIO);
    }
    else
    {
        originalFaceMat = Mat(16,16,CV_8UC3);
    }
    return originalFaceMat;
}
Rect faceDetector::getFaceRect()
{

    Rect face = FaceRIO;
    if(isFaceDetected())
    {
        face.x = cvRound(face.x*widthScale);
        face.y = cvRound(face.y*heightScale);
        face.width = cvRound(face.width*widthScale);
        face.height = cvRound(face.height*heightScale);
    }
    return face;
}


void faceDetector::detectLeftEye()
{
    /*if(EyeEnable)
    {
        if(isFaceDetected())
        {
            Rect eyeRect = Rect(0,0,FaceRIO.width/2,FaceRIO.height/2);
            Mat eyesMat = faceMat(eyeRect);
            detectObject(eyesClassifier,eyesMat,leftEyes);

            if(leftEyes.size() == 0)
            {
                if(GlassEyeEnable)
                {
                    detectObject(glassEyesClassifier,eyesMat,leftEyes);
                }
            }
        }
    }*/
    if(isFaceDetected())
    {
        detectLeftEye(faceMat,leftEyes);
    }
}


bool faceDetector::detectLeftEye(Mat & faceMat,vector<Rect> &leftEyes)
{
    leftEyes.clear();
    if(EyeEnable)
    {
        Rect eyeRect = Rect(0,0,faceMat.cols/2,faceMat.rows/2);
        Mat eyesMat = faceMat(eyeRect);
        detectLargestObject(eyesClassifier,eyesMat,leftEyes);
        if(leftEyes.size() == 0)
        {
            if(GlassEyeEnable)
            {
                detectLargestObject(glassEyesClassifier,eyesMat,leftEyes);
            }
        }

    }

    return (leftEyes.size() > 0);
}

Rect faceDetector::getLeftEyeRect()
{
    Rect leftEye = getRelativeLEyeRect();
    if(leftEye != Rect(-1,-1,-1,-1))
    {
        leftEye = leftEyes[0];
        leftEye.x = (FaceRIO.x + leftEye.x)*widthScale;
        leftEye.y = (FaceRIO.y + leftEye.y)*heightScale;
        leftEye.width = leftEye.width*widthScale;
        leftEye.height = leftEye.height*widthScale;
    }
    return leftEye;
}

Rect faceDetector::getRelativeLEyeRect()
{
    Rect leftEye = Rect(-1,-1,-1,-1);
    if(leftEyes.size() > 0)
    {
        leftEye = leftEyes[0];
    }

    return leftEye;
}



void faceDetector::detectRightEye()
{
    if(isFaceDetected())
    {
        detectRightEye(faceMat,rightEyes);
    }
}

bool faceDetector::detectRightEye(Mat & faceMat,vector<Rect> &rightEyes)
{
    rightEyes.clear();
    if(EyeEnable)
    {
        Rect eyeRect = Rect(faceMat.cols/2,0,faceMat.cols/2,faceMat.rows/2);
        Mat eyesMat = faceMat(eyeRect);
        detectLargestObject(eyesClassifier,eyesMat,rightEyes);
        if(rightEyes.size() == 0)
        {
            if(GlassEyeEnable)
            {
                detectLargestObject(glassEyesClassifier,eyesMat,rightEyes);
            }
        }

    }

    return (rightEyes.size() > 0);

}

Rect faceDetector::getRightEyeRect()
{
    Rect rightEye = getRelativeREyeRect();
    if(rightEye != Rect(-1,-1,-1,-1))
    {
        rightEye = rightEyes[0];
        rightEye.x = (FaceRIO.x + rightEye.x + FaceRIO.width/2)*widthScale;
        rightEye.y = (FaceRIO.y + rightEye.y)*heightScale;
        rightEye.width = rightEye.width*widthScale;
        rightEye.height = rightEye.height*widthScale;

    }
    return rightEye;
}

Rect faceDetector::getRelativeREyeRect()
{
    Rect rightEye = Rect(-1,-1,-1,-1);
    if(rightEyes.size() > 0)
    {
        rightEye = rightEyes[0];
        rightEye.x += FaceRIO.width/2;
    }
    return rightEye;
}

void faceDetector::detectNose()
{
    if(NoseEnable)
    {
        Mat noseMat;
        if(isFaceDetected())
        {
            Rect noseRect = Rect(0,FaceRIO.height/4,FaceRIO.width,FaceRIO.height/2);
            noseMat = faceMat(noseRect);
            detectLargestObject(noseClassifier,noseMat,noses);
        }
    }
}


Rect faceDetector::getNoseRect()
{
    Rect noseRect = getRelateiveNoseRect();
    if(noseRect != Rect(-1,-1,-1,-1))
    {
        noseRect.x = (FaceRIO.x + noseRect.x)*widthScale;
        noseRect.y = (FaceRIO.y + noseRect.y)*heightScale;
        noseRect.width = noseRect.width*widthScale;
        noseRect.height = noseRect.height*widthScale;
    }
    return noseRect;
}

Rect faceDetector::getRelateiveNoseRect()
{
    Rect noseRect = Rect(-1,-1,-1,-1);

    if(noses.size() > 0)
    {
        noseRect = noses[0];
        noseRect.y += FaceRIO.height/4;
    }

    return noseRect;
}

void faceDetector::detectMouth()
{
    if(MouthEnable)
    {
        if(isFaceDetected())
        {
            Rect mouthRect = Rect(0,FaceRIO.height/2,FaceRIO.width,FaceRIO.height/2);
            Mat mouthMat = faceMat(mouthRect);
            detectLargestObject(mouthCalssifier,mouthMat,mouth);
        }

    }
}

Rect faceDetector::getMouthRect()
{
    Rect mouthRect = getRelativeMouthRect();
    if(mouthRect != Rect(-1,-1,-1,-1))
    {
        mouthRect.x = (FaceRIO.x + mouthRect.x)*widthScale;
        mouthRect.y = (FaceRIO.y + mouthRect.y)*heightScale;
        mouthRect.width = mouthRect.width*widthScale;
        mouthRect.height = mouthRect.height*widthScale;
    }
    return mouthRect;
}

Rect faceDetector::getRelativeMouthRect()
{
    Rect mouthRect = Rect(-1,-1,-1,-1);

    if(mouth.size() > 0)
    {
        mouthRect = mouth[0];
        mouthRect.y += FaceRIO.height/2;
    }

    return mouthRect;
}

void faceDetector::detectObject(CascadeClassifier &classifier, const Mat &input,vector<Rect> & objects)
{
    int miniNeighbor = 4;
    Size miniFeatureSize = Size(objSize,objSize);
    float scaleFactor = 1.1f;
    int flag = CV_HAAR_SCALE_IMAGE;
    classifier.detectMultiScale(input,objects,scaleFactor,miniNeighbor,0|flag,miniFeatureSize);
}

void faceDetector::detectLargestObject(CascadeClassifier & classifier, const Mat & input,vector<Rect> & outPut)
{
    int minNeighbor = 4;
    Size miniFeature = Size(objSize/3,objSize/3);
    float scaleFactor = 1.1f;
    int Flag = CV_HAAR_FIND_BIGGEST_OBJECT;
    classifier.detectMultiScale(input,outPut,scaleFactor,minNeighbor,0|Flag,miniFeature);

}
