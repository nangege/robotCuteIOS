//
//  imageViewController.h
//  opencvProject
//
//  Created by zhanghuiping on 14-1-10.
//  Copyright (c) 2014年 Nobody. All rights reserved.
//
#import <UIKit/UIKit.h>
#import "CSScreenRecorder.h"
#import "CSRecordingListViewController.h"

@interface imageViewController : UIViewController<CSScreenRecorderDelegate>

@property (weak, nonatomic) IBOutlet UIImageView *imageView;

@property (weak, nonatomic) IBOutlet UIImageView *faceView;

@property (weak, nonatomic) IBOutlet UILabel *messageLabel;

@property (weak, nonatomic) IBOutlet UIButton *startButton;

@property (weak, nonatomic) IBOutlet UIButton *stopButton;

@property (weak, nonatomic) IBOutlet UIButton *collectButton;


@end
