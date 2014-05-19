//
//  imageViewController.m
//  opencvProject
//
//  Created by zhanghuiping on 14-1-10.
//  Copyright (c) 2014年 Nobody. All rights reserved.
//


#import "UIImage+OpenCV.h"
#import <dispatch/dispatch.h>
#import "imageViewController.h"
#include "facesketch.h"
#include "facedetector.h"
#include "facerecognition.h"
#include "cartoon.h"

@interface imageViewController ()
{
    cv::VideoCapture *_videoCapture;
    cv::Mat _lastFrame;
    faceSketch * sketchedFace;
    faceDetector *  detect;
    faceRecognition * faceRec;
    
    CSScreenRecorder * recoeder;
}

@end

@implementation imageViewController

- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
    self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
    if (self)
    {
        // Custom initialization
    
        
    }
    return self;
}
- (IBAction)captureImage:(id)sender
{
    static bool runFlag = false;
    
    runFlag = !runFlag;
    if(runFlag)
    {
        _videoCapture->open(1);
        [self.startButton setTitle:@"停止" forState:UIControlStateSelected];
        [self.startButton setTitle:@"停止" forState:UIControlStateNormal];
        
    }
    else
    {
        _videoCapture->release();
        [self.startButton setTitle:@"开始" forState:UIControlStateSelected];
        [self.startButton setTitle:@"开始" forState:UIControlStateNormal];
    }
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        int time  = 0;



        while(runFlag)
        {
            @autoreleasepool
            {
                if (_videoCapture && _videoCapture->grab())
                {
                    time = time + 1;
                    (*_videoCapture) >> _lastFrame;
                    cv::resize(_lastFrame, _lastFrame, cv::Size(180,240));
                    cv::cvtColor(_lastFrame, _lastFrame, CV_BGR2RGB);
                    cv::flip(_lastFrame, _lastFrame, 1);
        
                    sketchedFace->reset();
                    
                    {
                        detect->setMatToDetect(_lastFrame);
                        detect->detectAllFeatures();
                        
                        if(detect->isFaceDetected())
                        {
                            cv::Rect face = detect->getFaceRect();
                            rectangle(_lastFrame,face,Scalar(255,100,0),1);
                            
                            cv::Rect leftEye = detect->getLeftEyeRect();
                            rectangle(_lastFrame,leftEye,Scalar(0,255,30),1);
                            circle(_lastFrame,cv::Point(leftEye.x + leftEye.width/2,leftEye.y + leftEye.height/2),
                                   10,Scalar(100,0,200),1);
                            
                            cv::Rect rightEye = detect->getRightEyeRect();
                            rectangle(_lastFrame,rightEye,Scalar(0,0,255),1);
                            circle(_lastFrame,cv::Point(rightEye.x + rightEye.width/2,rightEye.y + rightEye.height/2),
                                   10,Scalar(100,0,200),1);
                            
                            cv::Rect nose = detect->getNoseRect();
                            rectangle(_lastFrame,nose,Scalar(77,200,40),1);
                            
                            cv::Rect mouth = detect->getMouthRect();
                            rectangle(_lastFrame,mouth,Scalar(200,10,70),1);
                        }
                        
                        if(detect->isFaceDetected())
                        {
                            cv::Rect face = detect->getFaceRect();
                            sketchedFace->setTrack();
                            sketchedFace->setTrackPoint(face.x/(180.0  - face.width),face.y/(240.0 - face.height));
                            time = 0;
                            
                            if(faceRec->isTrained())
                            {
                                [self recognizeFace];
                            }
                            else
                            {
                                [self collectFace];
                            }
                        }
                        else
                        {
                            time ++;
                            if(time > 150)
                            {
                                sketchedFace->setSleep();
                            }
                            else if (time > 40)
                            {
                                sketchedFace->setSmile();
                            }
                        }
                    }
                    
                    Mat showmat(_lastFrame.size(),CV_8UC1);
                    Mat showMat = Mat(_lastFrame.size(), CV_8UC3);
                    
                    sketchedFace->sketchWholeFace();
                    UIImage * image = [UIImage imageWithCVMat:_lastFrame];
                    UIImage * faceImage = [UIImage imageWithCVMat:sketchedFace->getSketchedMat()];
                    
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [self.imageView setImage:image];
                        [self.faceView setImage:faceImage];
                    });
                    
                }
                else
                {
                    NSLog(@"Failed to grab frame");
                }
            }
        
        }
        
    });

}

- (void ) startRecoder
{
    [[NSFileManager defaultManager] removeItemAtPath:[self inDocumentsDirectory:@"video.mp4"] error:nil];
    
    NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
    [dateFormatter setDateFormat:@"MM:dd:yyyy h:mm:ss a"];
    NSString *date = [dateFormatter stringFromDate:[NSDate date]];
    NSString *outName = [NSString stringWithFormat:@"%@.mp4", date];
    NSString *videoPath = [self inDocumentsDirectory:outName];
    
    // Set the number of audio channels
    NSNumber *audioChannels = [[NSUserDefaults standardUserDefaults] objectForKey:@"channels"];
    NSNumber *sampleRate = [[NSUserDefaults standardUserDefaults] objectForKey:@"samplerate"];
    NSString *audioPath = [self inDocumentsDirectory:@"audio.caf"];
    
    recoeder.videoOutPath = videoPath;
    recoeder.audioOutPath = audioPath;
    recoeder.numberOfAudioChannels = audioChannels;
    recoeder.audioSampleRate = sampleRate;
    
    [recoeder startRecordingScreen];
}

- (void) collectFace
{
    static int labelNum = 0;
    if(detect->isFaceDetected()
       && detect->getLeftEyeRect() != cv::Rect(-1,-1,-1,-1)
       && detect->getRightEyeRect() != cv::Rect(-1,-1,-1,-1))
    {
        Mat faceMat = detect->getFaceMat();
        faceRec->addFace(faceMat,labelNum);
        if(faceRec->hasEnoughDataToTrain())
        {
            dispatch_async(dispatch_get_main_queue(), ^{
                [self.messageLabel setText:@"trainModule"];
            });
            faceRec->trainFacesMat();
            labelNum ++;
        }
    }
    
}

- (void) recognizeFace
{
    if(detect->isFaceDetected()
       && detect->getLeftEyeRect() != cv::Rect(-1,-1,-1,-1)
       && detect->getRightEyeRect() != cv::Rect(-1,-1,-1,-1))
    {
        Mat faceMat = detect->getFaceMat();
        
        if(faceRec->isTrained())
        {
            int label = faceRec->predictFaceMat(faceMat);
            if(label == -1)
            {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self.messageLabel setText:@"unknown person"];
                });
            }
            else
            {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self.messageLabel setText:@"known person"];
                });
            }

        }
    }
}

- (void )processFrame
{
    
}

- (void)viewDidLoad
{
    [super viewDidLoad];
    // Do any additional setup after loading the view from its nib.
    _videoCapture = new cv::VideoCapture(1);
    sketchedFace = new faceSketch();
    detect = new faceDetector();
    faceRec = new faceRecognition();
    
    NSLog(@"%@",[[NSBundle mainBundle] bundlePath]);
    string preString = [[[NSBundle mainBundle] bundlePath] UTF8String];
    detect->setpreString(preString);
    detect->initClassifier();
    
    sketchedFace->reset();
    sketchedFace->sketchWholeFace();
    UIImage * image = [UIImage imageWithCVMat:sketchedFace->getSketchedMat()];
    [self.faceView setImage:image];
    
    if (!_videoCapture->open(CV_CAP_AVFOUNDATION))
    {
        NSLog(@"Failed to open video camera");
    }
    
    self.startButton.layer.borderColor = [[UIColor greenColor] CGColor];
    self.stopButton.layer.borderColor = [[UIColor greenColor] CGColor];
    self.collectButton.layer.borderColor = [[UIColor greenColor] CGColor];
    
    self.startButton.layer.cornerRadius = 30.0f;
    self.stopButton.layer.cornerRadius = 30.0f;
    self.collectButton.layer.cornerRadius = 30.0f;
    
    self.startButton.layer.borderWidth = 2.0f;
    self.stopButton.layer.borderWidth = 2.0f;
    self.collectButton.layer.borderWidth = 2.0f;
    
    self.imageView.layer.borderWidth = 4.0f;
    self.imageView.layer.borderColor = [[UIColor whiteColor] CGColor];
    
    recoeder = [[CSScreenRecorder alloc] init];
    recoeder.delegate = self;
    
    [self.collectButton addTarget:self action:@selector(collectPressed) forControlEvents:UIControlEventTouchUpInside];
    [self.navigationController.navigationBar setHidden:YES];
    
}


- (void)viewWillAppear:(BOOL)animated
{
    [self.navigationController.navigationBar setHidden:YES];
}
- (void) collectPressed
{    
    [self.navigationController pushViewController:[[CSRecordingListViewController alloc] init] animated:YES];
}


- (IBAction)stopRecorder:(id)sender {
    static bool recorderFlag = false;
    
    recorderFlag = !recorderFlag;
    
    if(recorderFlag)
    {
        [self startRecoder];
    }
    else
    {
        [recoeder stopRecordingScreen];
        [self.stopButton setTitle:@"录像" forState:UIControlStateNormal];
        [self.stopButton setTitle:@"录像" forState:UIControlStateSelected];
    }
    
}

- (NSString *)inDocumentsDirectory:(NSString *)path {
	NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
	NSString *documentsDirectory = [paths objectAtIndex:0];
	return [documentsDirectory stringByAppendingPathComponent:path];
}

#pragma recorder
- (void)screenRecorderDidStopRecording:(CSScreenRecorder *)recorder
{
    
    
    
}

- (void)screenRecorder:(CSScreenRecorder *)recorder recordingTimeChanged:(NSTimeInterval)recordingInterval
{
    NSDate *timerDate = [NSDate dateWithTimeIntervalSince1970:recordingInterval];
    
    // Make a date formatter (Possibly reuse instead of creating each time)
    NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
    [dateFormatter setDateFormat:@"mm:ss"];
    [dateFormatter setTimeZone:[NSTimeZone timeZoneForSecondsFromGMT:0.0]];
    
    // Set the current time since recording began
    NSString *timeString = [dateFormatter stringFromDate:timerDate];
    [self.stopButton setTitle:timeString forState:UIControlStateNormal];
}

// Stubs. Do Error handling shit here.
- (void)screenRecorder:(CSScreenRecorder *)recorder videoContextSetupFailedWithError:(NSError *)error
{
    
}

- (void)screenRecorder:(CSScreenRecorder *)recorder audioRecorderSetupFailedWithError:(NSError *)error
{
    
}

- (void)screenRecorder:(CSScreenRecorder *)recorder audioSessionSetupFailedWithError:(NSError *)error
{
    
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
