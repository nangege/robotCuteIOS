#include "facesketch.h"

faceSketch::faceSketch():leftAngle(0),rightAngle(0),eyeScale(1.0),MOVE_EYE_BALL(false),L_Ball_Angle(0),R_Ball_Angle(0),
UPBOUND(1),LOWBOUND(-1),CHANGE_SPEED(0.1),blinkFlag(false),mouthAngle(0)
{

}

void faceSketch::showFace()
{
    if(sketchedMat.data)
    {
        imshow("sketcked Face",sketchedMat);
    }
}

void faceSketch::setEyeScale(float scale)
{
    eyeScale = scale;
    if( eyeScale < 0 )
    {
        eyeScale = 0;
    }
    else if(eyeScale > 1.5)
    {
        eyeScale = 1.5;
    }
}

void faceSketch::reset()
{
    sketchedMat.create(Size(640,480),CV_8UC3);
    vector<Mat> rgb;
    split(sketchedMat,rgb);
    for(int i = 0 ; i < sketchedMat.cols;  ++i)
    {
        for(int j = 0 ; j < sketchedMat.rows; ++j)
        {
            //rgb[0].at<uchar>(j,i) = 255;
            //rgb[1].at<uchar>(j,i) = 100;
            //rgb[2].at<uchar>(j,i) = 200;
            rgb[0].at<uchar>(j,i) = 0;
            rgb[1].at<uchar>(j,i) = 204;
            rgb[2].at<uchar>(j,i) = 153;
        }
    }
    merge(rgb,sketchedMat);
}

void faceSketch::setEyeAngle(int lEye, int rEye)
{
    leftAngle = lEye;
    rightAngle = rEye;
}

void faceSketch::setAngery()
{
    setEyeAngle(30,-30);
    setMouthAngle(180);
}

void faceSketch::setLovely()
{
    setEyeAngle(30,30);
    setMouthAngle(20);
}

void faceSketch::setNormal()
{
    setEyeAngle(0,0);
    setMouthAngle(0);
}


void faceSketch::setBound(float low, float up)
{
    LOWBOUND = low;
    UPBOUND  = up;
}

void faceSketch::setTrack()
{
    setChangeSpeed(0);
    setBound(0,0.8);
    this->setNormal();
    this->setMoveEyeBall(true);
    this->setBlink(false);
    this->setEyeScale(1.0);
}

void faceSketch::setAnger()
{
    setChangeSpeed(0.3);
    setBound(0,4);
    this->setAngery();
    this->setMoveEyeBall(false);
    this->setBlink(true);
}

void faceSketch::setSmile()
{
    setChangeSpeed(0.6);
    setBound(0,5);
    this->setNormal();
    this->setMoveEyeBall(false);
    this->setBlink(true);
}

void faceSketch::setSleep()
{
    setChangeSpeed(0.1);
    setBound(-0.6,0.5);
    this->setNormal();
    this->setMoveEyeBall(false);
    this->setBlink(true);
}


void faceSketch::setTrackPoint(float x_ratio, float y_ratio)
{
    float L_Angle = 0;
    float R_Angle = 0;
    
    float L_X_Offset = x_ratio - LEFT_EYE_X;
    float L_Y_Offset = y_ratio - EYE_Y;
    
    if(L_X_Offset == 0.0)
    {
        if(L_Y_Offset > 0)
        {
            L_Angle = 90;
        }
        else
        {
            L_Angle = -90;
        }
    }
    else
    {
        L_Angle = atan(L_Y_Offset/L_X_Offset);
        if(L_X_Offset < 0)
        {
            L_Angle += CV_PI;
        }
        
    }
    
    float R_X_Offset = x_ratio - RIGHT_EYE_X;
    float R_Y_Offset = y_ratio - EYE_Y;
    if(R_X_Offset == 0.0)
    {
        if(R_Y_Offset < 0)
        {
            R_Angle = 90;
        }
        else
        {
            R_Angle = -90;
        }
    }
    else
    {
        R_Angle = atan(R_Y_Offset/R_X_Offset);
        if(R_X_Offset < 0)
        {
            R_Angle += CV_PI;
        }
    }
    
    this->setEyeBall(L_Angle,R_Angle);
    
}

void faceSketch::sketchWholeFace()
{
    static float scale = 1.0;
    static bool eyeFlag = false;
    static int  eyeAngleScale = 1;
    
    if(blinkFlag)
    {
        if(eyeFlag)
        {
            scale -= CHANGE_SPEED;
            if(scale <= LOWBOUND)
            {
                eyeFlag = 0;
            }
        }
        else
        {
            scale += CHANGE_SPEED;
            if(scale >= UPBOUND)
            {
                eyeFlag = 1;
            }
        }
        
        this->setEyeScale(scale);
    }
    
    int width = sketchedMat.cols;
    int height = sketchedMat.rows;

    int L_EYE_WIDTH = width*LEFT_EYE_X;
    int R_EYE_WIDTH = width*RIGHT_EYE_X;
    int EYE_HEIGHT  = EYE_Y * height;

    ellipse(sketchedMat,Point(L_EYE_WIDTH,20),Size(50,15),180,0,180,Scalar(255,255,255),2);  //眉毛
    ellipse(sketchedMat,Point(R_EYE_WIDTH,20),Size(50,15),180,0,180,Scalar(255,255,255),2);

    ellipse(sketchedMat,Point(L_EYE_WIDTH,EYE_HEIGHT),Size(80,80*eyeScale),leftAngle,0,360,Scalar(255,255,255),-1);//眼眶
    ellipse(sketchedMat,Point(R_EYE_WIDTH,EYE_HEIGHT),Size(80,80*eyeScale),rightAngle,0,360,Scalar(255,255,255),-1);

    float eyeSize = 30;
    if(eyeScale <= 0.5)
    {
        eyeSize = eyeSize * eyeScale/2;
    }
    //眼球
    if(MOVE_EYE_BALL)
    {
        int Move_Length = 80 - eyeSize;
        circle(sketchedMat,Point(L_EYE_WIDTH + Move_Length * cos(L_Ball_Angle),EYE_HEIGHT + Move_Length*sin(L_Ball_Angle)),
               eyeSize,Scalar::all(0),-1);
        circle(sketchedMat,Point(R_EYE_WIDTH + Move_Length * cos(R_Ball_Angle),EYE_HEIGHT + Move_Length*sin(R_Ball_Angle)),
               eyeSize,Scalar::all(0),-1);
    }
    else
    {
        circle(sketchedMat,Point(L_EYE_WIDTH,EYE_HEIGHT),eyeSize,Scalar::all(0),-1);
        circle(sketchedMat,Point(R_EYE_WIDTH,EYE_HEIGHT),eyeSize,Scalar::all(0),-1);
    }

    //嘴巴
    ellipse(sketchedMat,Point(width/2,height*0.8),Size(width/6,height/12),mouthAngle,0,180,Scalar(200,100,50),2);
}
