/*
Name: Harris Corner Detection
Author: Gong Lixue 3130104153@zju.edu.cn
Description: Without using relevant harris corner function in opencv to realize Harris Corner Detection algorithm.
			 The main part of this algorithm is self-completed except for some maxtrix calculation.
Date: 2015-12-19
*/


#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <stdlib.h>
using namespace std;
int main(int argc, char* argv[])
{
	IplImage* img_0 = cvLoadImage(argv[1]);
	IplImage* img_gray = cvCreateImage(cvGetSize(img_0), IPL_DEPTH_8U, 1);
	cvCvtColor(img_0, img_gray, CV_BGR2GRAY);  //转为单通道灰度
	float k = atof(argv[2]);
	IplImage* img = cvCreateImage(cvGetSize(img_0), 32, 1);
	cvConvertScale(img_gray, img);
	cvSmooth(img, img, CV_GAUSSIAN, 3, 3);

	IplImage* Ix = cvCreateImage(cvGetSize(img), 32, 1);
	IplImage* Iy = cvCreateImage(cvGetSize(img), 32, 1);
	// sobel算子滤波，得到Ix,Iy一阶导数
	int aperture_size = atoi(argv[3]);
	cvSobel(img, Ix, 1, 0, aperture_size);
	cvSobel(img, Iy, 0, 1, aperture_size);

	// 计算每个像素位置对应的x和y方向梯度的乘积，即计算Ixy ,Ixx, Iyy
	// 使用函数cvMul计算同size矩阵的对应位置的元素级乘积
	IplImage* Ixx = cvCreateImage(cvGetSize(img), 32, 1);
	IplImage* Ixy = cvCreateImage(cvGetSize(img), 32, 1);
	IplImage* Iyy = cvCreateImage(cvGetSize(img), 32, 1);
	cvMul(Ix, Iy, Ixy);
	cvMul(Ix, Ix, Ixx);
	cvMul(Iy, Iy, Iyy);

	
	//构造R
	IplImage* R = cvCreateImage(cvGetSize(img), 32, 1);
	IplImage* lamdaMin = cvCreateImage(cvGetSize(img), 32, 1);  //最小特征图
	IplImage* lamdaMax = cvCreateImage(cvGetSize(img), 32, 1);  //最大特征图


	int width = cvGetSize(img).width;
	int height = cvGetSize(img).height;

	/*构造M，并计算R*/	
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			CvMat* M = cvCreateMat(2, 2, CV_64FC1);   //对每个像素 构造一个2*2大小的M
		
			float m1, m2, m4;
			m1 = cvGetReal2D(Ixx, i, j);
			m2 = cvGetReal2D(Ixy, i, j);
			m4 = cvGetReal2D(Iyy, i, j);
			cvSetReal2D(M, 0, 0, m1);
			cvSetReal2D(M, 0, 1, m2);
			cvSetReal2D(M, 1, 0, m2);
			cvSetReal2D(M, 1, 1, m4);

			//求M特征值
			CvMat* ProVector = cvCreateMat(2, 2, CV_64FC1);  //特征向量矩阵
			CvMat* ProValue = cvCreateMat(2, 1, CV_64FC1);  //特征值矩阵
			cvEigenVV(M, ProVector, ProValue);
			float v1 = cvmGet(ProValue, 0, 0);
			float v2 = cvmGet(ProValue, 1, 0);

			float r = v1*v2 - k*(v1 + v2)*(v1 + v2);   //用M的特征值计算R
			//cout << "r:" << r << "    ";
			cvSetReal2D(R, i, j, fabs(r));

			//设置最大 最小特征图
			cvSetReal2D(lamdaMax, i, j, v1);
			cvSetReal2D(lamdaMin, i, j, fabs(v2));
			
			//cout << "max:" << v1 << "min:" << v2 << endl;
		}
	}

	double max,min;
	cvMinMaxLoc(R, &min, &max);	

	//将R映射到0-255	
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++){
			float value = cvGetReal2D(R, i, j);
			value = (value - min) * 255 / (max - min);
			cvSetReal2D(R, i, j, value);
		}
	}
	cvMinMaxLoc(R, &min, &max);
	cout << " max :" << max;
	cout << " min :" << min << endl;

	//将最大特征图映射到0-255范围内
	double lamdaMax_max, lamdaMax_min;
	double lamdaMin_max, lamdaMin_min;
	cvMinMaxLoc(lamdaMax, &lamdaMax_min, &lamdaMax_max);
	cvMinMaxLoc(lamdaMin, &lamdaMin_min, &lamdaMin_max);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++){
			float v1 = cvGetReal2D(lamdaMax, i, j);
			float v2 = cvGetReal2D(lamdaMin, i, j);
			cvSetReal2D(lamdaMax, i, j, (v1 - lamdaMax_min) * 255 / (lamdaMax_max - lamdaMax_min));
			cvSetReal2D(lamdaMin, i, j, (v2 - lamdaMin_min) * 255 / (lamdaMin_max - lamdaMin_min));

		}
	}

	/* 比较R的8领域*/	
	for (int i = 1; i < height - 1; i++){
		for (int j = 1; j < width - 1; j++){
			//和8领域比较
			double center = cvGetReal2D(R,i,j);
			// 大于阈值
			//cout << "center:" << center;
			if (center > max/8){

				if (center>cvGetReal2D(R, i - 1, j - 1) && center > cvGetReal2D(R, i - 1, j) && center > cvGetReal2D(R, i-1, j + 1)
					&& center > cvGetReal2D(R, i, j - 1) && center >cvGetReal2D(R, i, j + 1)
					&& center > cvGetReal2D(R, i + 1, j - 1) && center > cvGetReal2D(R, i + 1, j) && center > cvGetReal2D(R, i+1, j+1))
				{
					cout << center << endl;
					cvCircle(img_0, CvPoint(j,i), 5, CvScalar(0, 0, 255), 1);
				}
			}
		}
	}
	
	/**/

	/* 比较cim 的领域
	for (int i = 1; i < height - 1; i++){
		int* ptr = (int*)cim->imageData + i*cim->width;
		int* ptr_pre = (int*)cim->imageData + (i - 1)*cim->width;
		int* ptr_suc = (int*)cim->imageData + (i + 1)*cim->width;
		for (int j = 1; j < width - 1; j++){
			//和8领域比较
			int center = ptr[j];
			// 大于阈值
			//cout << "center:" << center;
			if (center > 0){

				if (center>ptr_pre[j - 1] && center > ptr_pre[j] && center > ptr_pre[j + 1]
					&& center > ptr[j - 1] && center > ptr[j + 1]
					&& center > ptr_suc[j - 1] && center > ptr_suc[j] && center > ptr_suc[j + 1])
				{
					cout << center << endl;
					cvCircle(img, CvPoint(j, i), 5, CvScalar(0, 0, 255), 1);
					//cvCircle(R, CvPoint(j, i), 5, CvScalar(0, 0, 255), 1);
				}
			}
		}
	}
	*/


	cvNamedWindow("src_HW2_3130104153");
	cvShowImage("src_HW2_3130104153", img_0);
	cvNamedWindow("R_HW2_3130104153");
	cvShowImage("R_HW2_3130104153", R);
	cvNamedWindow("max_HW2_3130104153");
	cvShowImage("max_HW2_3130104153", lamdaMax);
	cvNamedWindow("min_HW2_3130104153");
	cvShowImage("min_HW2_3130104153", lamdaMin);

	//保存图像
	cvSaveImage("./output/R.jpg", R);
	cvSaveImage("./output/lamdaMax.bmp", lamdaMax);
	cvSaveImage("./output/lamdaMin.bmp", lamdaMin);
	cvSaveImage("./output/corner_detection.jpg", img_0);
	

	cvWaitKey(0);
	cvReleaseImage(&img);
	cvReleaseImage(&img_0);
	cvReleaseImage(&img_gray);
	cvReleaseImage(&Ix);
	cvReleaseImage(&Iy);
	cvReleaseImage(&Ixx);
	cvReleaseImage(&Iyy);
	cvReleaseImage(&Ixy);
	cvReleaseImage(&R);
	cvReleaseImage(&lamdaMax);
	cvReleaseImage(&lamdaMin);
	//cvReleaseImage(&IxyIxy);
	//cvReleaseImage(&R_gray);

	return 0;

}

/*
Name: Harris Corner Detection
Author: Gong Lixue 3130104153@zju.edu.cn
Description: Without using relevant harris corner function in opencv to realize Harris Corner Detection algorithm.
The main part of this algorithm is self-completed except for some maxtrix calculation.
Date: 2015-12-19
*/
