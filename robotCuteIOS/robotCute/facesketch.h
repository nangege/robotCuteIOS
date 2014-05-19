#ifndef FACESKETCH_H
#define FACESKETCH_H

#include <iostream>
using namespace std;
using namespace cv;


const float LEFT_EYE_X  = 0.26;
const float RIGHT_EYE_X = 1 - LEFT_EYE_X;
const float EYE_Y       = 0.35;
class faceSketch
{
public:
    faceSketch();

    Mat getSketchedMat(){return sketchedMat;}

    void sketchWholeFace();

    void reset();

    void setEyeAngle(int lEye,int rEye);

    void setMouthAngle(int angle){mouthAngle = angle;}
    void setMoveEyeBall(bool flag){MOVE_EYE_BALL = flag;}

    void setEyeScale(float scale);

    void showFace();

    void setEyeBall(float L,float R){L_Ball_Angle = L; R_Ball_Angle = R;}
    void setTrackPoint(float x_ratio, float y_ratio);
    
    void setBlink(bool flag){blinkFlag = flag;}
    
    void setBound(float low, float up);
    void setTrack();
    void setAnger();
    void setSmile();
    void setSleep();
    void setChangeSpeed(float speed){CHANGE_SPEED = speed;}

private:
    Mat sketchedMat;
    Mat faceMat;
    RNG rng;

    int leftAngle ;
    int rightAngle;

    int mouthAngle;

    float eyeScale;

    bool MOVE_EYE_BALL;

    float L_Ball_Angle;
    float R_Ball_Angle;
    
    float CHANGE_SPEED;
    float UPBOUND ;
    float LOWBOUND ;
    
    bool blinkFlag;
    
    void setAngery();
    
    void setLovely();
    
    void setNormal();
};

#endif // FACESKETCH_H
