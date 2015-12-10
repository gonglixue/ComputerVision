#include "cv.h"
#include "highgui.h"
#include <string.h>
#include <cstring>
#include <iostream>
using namespace std;

void rotateImage(IplImage* img, IplImage *img_rotate, int degree)
{
	CvPoint2D32f center;
	
	center.x = float(img->width / 2.0 + 0.5);
	center.y = float(img->height / 2.0 + 0.5);
	//计算二维旋转的仿射变换矩阵
	float m[6];
	CvMat M = cvMat(2, 3, CV_32F, m);
	cv2DRotationMatrix(center, degree, 1, &M);

	cvWarpAffine(img, img_rotate, &M, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
}

CvScalar random_color(CvRNG *rng)
{
	int icolor = cvRandInt(rng);
	return CV_RGB(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

int main(int argc, char *argv[])
{
	char Video[30];
	char*s = "\\video.avi";
	strcpy(Video, argv[1]);
	strcat(Video, s);
	CvCapture* capture = 0;
	capture = cvCreateFileCapture(Video);
	cout << Video;
	if (!capture)
		return -1;

	IplImage *bgr_frame = cvQueryFrame(capture);
	double fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
	CvSize size = cvSize((int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH), (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT));
	int W = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
	int H = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
	IplImage* photo;
	IplImage* resize_photo;
	IplImage* rotate_photo = cvCreateImage(size, IPL_DEPTH_8U, 3);

	char outputPath[30];
	strcpy(outputPath, argv[1]);
	strcat(outputPath, "/OUTPUT.avi");
	CvVideoWriter *writer = cvCreateVideoWriter(
		outputPath,
		CV_FOURCC('M', 'J', 'P', 'G'),
		fps,
		size
		);

	// 片头	
	CvSize diameter;
	CvRNG rng;
	diameter.width = diameter.height = 100;
	CvPoint center(diameter.width, H / 2);
	CvPoint center2(diameter.width + 30, H / 2 - 50);

	while (1)
	{
		IplImage* start_img = cvCreateImage(size, IPL_DEPTH_8U, 3);

		cvEllipse(start_img, center, diameter, 0, 30, 330, random_color(&rng), -1, 8, 0);	
		cvCircle(start_img, center2, 10, CV_RGB(0, 0, 0), -1, 8, 0);

		cvWaitKey(500);
		center.x += 100;  //移动圆心
		center2.x += 100;

		cvWriteFrame(writer, start_img);
		cvReleaseImage(&start_img);
		if (center.x >= W)
			break;

	}
	
	CvPoint textPos(W / 2 - 100, H - 100);
	CvFont font;
	double hScale = 1.0;
	double vScale = 1.0;
	int lineWidth = 3;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, hScale, vScale, 0, lineWidth);

	//读入图片，从文件夹遍历？
	char filename[10];
	for (int i = 0; i < 50; i++)
	{
		sprintf(filename, "/%d.jpg", i/10);
		char imgPath[30];
		strcpy(imgPath, argv[1]);
		strcat(imgPath, filename);
		photo = cvLoadImage(imgPath);
		resize_photo = cvCreateImage(size, IPL_DEPTH_8U, 3);
		cvResize(photo, resize_photo, CV_INTER_LINEAR);  //改变图片大小
		rotateImage(resize_photo, rotate_photo, (i%10)*18);
		cvPutText(rotate_photo, "3130104153 Gong Lixue", cvPoint(400, 600), &font, cvScalar(255, 255, 255));  //写入字幕
		cvWriteFrame(writer, rotate_photo);
	}

	while ((bgr_frame = cvQueryFrame(capture)) != NULL){
		cvPutText(bgr_frame, "3130104153 Gong Lixue", cvPoint(400,600), &font, cvScalar(0, 0, 0));
		cvWriteFrame(writer, bgr_frame);
	}

	cvReleaseVideoWriter(&writer);
	cvReleaseImage(&photo);
	cvReleaseImage(&resize_photo);
	cvReleaseCapture(&capture);
	cvReleaseImage(&bgr_frame);
	cvReleaseImage(&rotate_photo);

	cvDestroyWindow("test");
	return (0);

}
