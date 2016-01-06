/*
Name: EigenFace_train
Author: Gong Lixue 3130104153@zju.edu.cn
Description: Without using relevant eigenface function in opencv to realize EigenFace algorithm based on PCA.
			 The main part of this algorithm is self-completed except for some maxtrix calculation.
Date: 2016-01-06
*/


#include <cv.h>
#include <highgui.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;
IplImage* cutRoi(int id, int rows, int cols)
{
	char imagefile[25];
	char eyefile[25];
	sprintf(imagefile, "./train2/BioID_%04d.pgm", id);
	sprintf(eyefile, "./train2/BioID_%04d.eye", id);
	cout << imagefile << endl;
	cout << eyefile << endl;
	IplImage* image = cvLoadImage(imagefile);
	ifstream inf(eyefile);

	char buf[15];
	inf.getline(buf, 15);  //指针指向下一行
	int lx, ly, rx, ry;  //眼睛坐标
	// 读入眼睛坐标
	inf >> lx;
	inf >> ly;
	inf >> rx;
	inf >> ry;
	// 定义roi的起始点、长宽
	int x = rx - (lx - rx) / 2;
	int y = ry - (lx - rx);
	int w = 3.9 * (lx - rx) / 2;
	int h = 2.8 * (lx - rx);
	cout << x << " " << y << " " << w << " " << h << endl;
	inf.close();
	// 以一定比例裁剪
	// 设置roi
	cvSetImageROI(image, cvRect(x, y, w, h));
	// 拷贝出roi
	IplImage* dst = cvCreateImage(cvSize(w, h), image->depth, image->nChannels);
	cvCopy(image, dst);

	// 缩放到统一大小
	IplImage* result = cvCreateImage(cvSize(cols, rows), image->depth, image->nChannels);
	cvResize(image, result);

	// 转成单通道灰度图像，最终将该灰度图返回
	IplImage* image_gray = cvCreateImage(cvGetSize(result), IPL_DEPTH_8U, 1);
	cvCvtColor(result, image_gray, CV_BGR2GRAY);

	//灰度均衡，直方图均衡化
	cvEqualizeHist(image_gray, image_gray);
	if (id == 1){
		cvShowImage("myself", image_gray);
	}
	return image_gray;
	
}

int main(int argc, char* argv[])
{
	int M=40;
	int cols, rows;
	int N; //单幅图片像素
	float power=atof(argv[1]);  //能量百分比
	char* file_eigenface = argv[2];
	char* file_traincoeff = argv[3];
	char* file_meanface = argv[4];
	//IplImage* temp = cvLoadImage("./train/s1/5.pgm");
	//rows = cvGetSize(temp).height;
	//cols = cvGetSize(temp).width;
	rows = 112;
	cols = 85;
	N = cols*rows;
	CvMat* S = cvCreateMat(N, M,CV_32FC1);
	
	if (argc != 5 || power<=0 || power>1){
		cout << "Wrong argument!" << endl;
		cout << "argv[1]: 0<power<=1\n"
			<< "argv[2]: file name, eg: eigenface.txt" << endl
			<< "argv[3]: file name, eg: traincoeff.txt" << endl
			<< "argv[4]: file name, eg: meanface.txt" << endl;
		return -1;
	}

	//循环载入训练图片，构建出N*M矩阵S
	for (int i = 1; i <= M; i++)
	{
		//char filename[50];
		//sprintf(filename, "./train/s%d/5.pgm", i);
		//IplImage* image;
		//image = cvLoadImage(filename);
		// 模板预处理

		IplImage* image_gray = cvCreateImage(cvSize(cols,rows), IPL_DEPTH_8U, 1); //单通道灰度
		//cvShowImage(filename, image); 每张图片导入正常
		//cvCvtColor(image, image_gray, CV_BGR2GRAY);  
		image_gray = cutRoi(i - 1, rows, cols);		

		//灰度均衡，直方图均衡化
		//cvEqualizeHist(image_gray, image_gray);

		CvMat* img_mat = cvCreateMat(rows, cols, CV_16UC1);  //载入图片的矩阵形式，便于reshape
		cvConvert(image_gray, img_mat);  //把图片转化为矩阵		
		//reshape拉伸
		CvMat* oneCol,mathdr;
		oneCol = cvReshape(img_mat, &mathdr, 1, N); //把矩阵reshape，oneCol是reshape后的N行列向量
		for (int j = 0; j < N; j++){
			int value = cvGetReal2D(oneCol, j, 0);
			cvSetReal2D(S, j, i-1, value);  //S的一列就是一张图片。
		}
	}

	//对S中列向量求和，进而求出平均人脸
	CvMat* mean_face = cvCreateMat(N, 1,CV_16UC1);
	cvSetZero(mean_face);
	for (int i = 0; i < M; i++){
		CvMat* temp = cvCreateMat(N,1,CV_16UC1);  //64位出错
		cvGetCol(S, temp, i);
		cvAdd(temp, mean_face, mean_face);
	}	
	//cvReduce(S, mean_face, 1, CV_REDUCE_SUM);  //合并成一列
	// 求均值,数乘1/M
	cvConvertScale(mean_face, mean_face, 1.0 / M);

	/*平均人脸测试*/
	IplImage* mean_show = cvCreateImage(cvSize(rows, cols),IPL_DEPTH_8U,1);
	IplImage mean_show_hdr;
	CvMat* mean_reshape = cvCreateMat(rows, cols, CV_8UC1);
	CvMat mean_hdr;
	mean_reshape = cvReshape(mean_face, &mean_hdr, 1, rows);  //测试输出meanface都是0
	mean_show = cvGetImage(&mean_hdr, &mean_show_hdr);
	//cvShowImage("mean", mean_show);
	cvSaveImage("./output/mean.jpg", mean_show);  

	for (int i = 0; i < M; i++){
		for (int j = 0; j < N; j++){
			int S_value = cvGetReal2D(S, j, i);
			int mean_value = cvGetReal2D(mean_face, j, 0);
			cvSetReal2D(S, j, i, S_value - mean_value);
		}
	}
	// 求S的转置矩阵
	CvMat* ST = cvCreateMat(M, N, CV_32FC1);
	cvTranspose(S, ST);
	//求协方差矩阵S'S，大小为MM
	CvMat* Col_mat = cvCreateMat(M, M,CV_32FC1);
	cvMatMul(ST, S, Col_mat);  //矩阵相乘
	//求矩阵Col_mat的特征值与特征向量，有M-1个特征向量？
	//构造输出特征向量矩阵
	CvMat* ProVector = cvCreateMat(M, M, CV_32FC1);
	//构造输出特征值矩阵
	CvMat* ProValue = cvCreateMat(M, 1, CV_32FC1);
	cvEigenVV(Col_mat, ProVector, ProValue, 1.0e-6F);
	//特征人脸空间
	CvMat* eigenface = cvCreateMat(N, int(M*power), CV_32FC1);
	//把ProVector的前 M*power列取出来
	CvMat* ProVector2 = cvCreateMat(M, int(M*power), CV_32FC1);
	for (int i = 0; i < int(M*power); i++){
		for (int j = 0; j < M; j++){
			float value = cvGetReal2D(ProVector, j, i);
			cvSetReal2D(ProVector2, j, i, value);
		}
	}
	//特征向量矩阵左乘S
	cvMatMul(S, ProVector2, eigenface);
	M = (int)M*power;
	
	cout << "----------"<<endl;
	//对eigenface矩阵每一列归一化
	CvMat* tempVector = cvCreateMat(N,M,CV_32FC1);
	cvPow(eigenface,tempVector,2);  //元素平方
	CvMat* temp2 = cvCreateMat(1,M,CV_32FC1);
	cvReduce(tempVector, temp2, 0, CV_REDUCE_SUM);  //平方后求和，合并成一行 ??? dim到底。。。
	for (int i = 0; i < M; i++){   //列
		double sum;
		sum = cvGetReal2D(temp2, 0, i); //第0行，第i列
		//cout << "sum:" << sum << endl;
		for (int j = 0; j < N; j++){    //行
			double value;
			value = cvGetReal2D(eigenface, j, i);
			value = value / (sqrt(sum));
			cvSetReal2D(eigenface, j, i, value);
			
		}
	}

	/*测试是不是真的归一化了*/
	/*
	double testsum = 0;
	for (int i = 0; i < N; i++){
		double value;
		value = cvGetReal2D(eigenface, i, 0);
		testsum += value*value;
	}
	testsum = sqrt(testsum);
	cout << " test sum:" << testsum;
	*/

	//eigenface空间构建完毕

	/*显示M张特征人脸，显示成功*/
	CvMat* FirstCol = cvCreateMat(N, 1, CV_32FC1);
	IplImage* Add10 = cvCreateImage(cvSize(cols, rows), IPL_DEPTH_8U, 1);
	//cvGetCol(eigenface, FirstCol, 0);  //改掉这个就对了。。。
	for (int k = 0; k < 10; k++){
		for (int i = 0; i < N; i++){
			float value = cvGetReal2D(eigenface, i, k);
			cvSetReal2D(FirstCol, i, 0, value);
		}
		// 将first_reshape映射在0-255
		cvNormalize(FirstCol, FirstCol, 255, 0, CV_MINMAX);
		CvMat* First_reshape = cvCreateMat(rows, cols, CV_32FC1);
		CvMat first_hdr;
		First_reshape = cvReshape(FirstCol, &first_hdr, 1, rows);
		IplImage* first_show = cvCreateImage(cvSize(cols, rows), IPL_DEPTH_8U, 1);
		IplImage first_show_hdr;
		first_show = cvGetImage(&first_hdr, &first_show_hdr);
		//cvShowImage("firstcol", first_show);
		char path[30];
		sprintf(path, "./output/col%d.jpg", k + 1);
		cvSaveImage(path, first_show);
		cvAddWeighted(Add10, 0.8, first_show, 0.2, 0, Add10);
		//cvAdd(Add10, first_show, Add10);
	}
	cvNormalize(Add10, Add10, 255, 0, CV_MINMAX);
	cvShowImage("Add10", Add10);
	cvSaveImage("./output/Add10.jpg", Add10);

	//计算训练集中的照片在基底上的投影坐标。
	CvMat* TrainCoeff = cvCreateMat(M, M, CV_32FC1);
	for (int i = 0; i < M; i++)
	{
		CvMat* temp = cvCreateMat(N, 1, CV_32FC1);
		cvGetCol(S, temp, i);
		CvMat* tempT = cvCreateMat(1, N, CV_32FC1);
		cvTranspose(temp, tempT);
		CvMat* c = cvCreateMat(1, M, CV_32FC1);
		cvMatMul(tempT, eigenface, c);  //和基底相乘
		CvMat* cT = cvCreateMat(M, 1, CV_32FC1);
		cvTranspose(c, cT);
		// 把cT加到TrainCoeff中
		for (int j = 0; j < M; j++){
			float value = cvGetReal2D(cT, j, 0);
			cvSetReal2D(TrainCoeff, j, i, value);
		}
	}

	//把基底空间eigenface 和 训练集照片的投影坐标traincoeff写入model文件
	cvSave(file_eigenface, eigenface);
	cvSave(file_traincoeff, TrainCoeff);
	cvSave(file_meanface, mean_face);

	//用前10张特征人脸（eigenface前10列）直接拼成一幅图像
	//前十张，每张映射到255，然后addWeight权重为0.1
	/*
	CvMat* add10 = cvCreateMat(N, 1, CV_32FC1);
	for (int i = 0; i < 1; i++){
		CvMat* colOfEigen = cvCreateMat(N, 1, CV_32FC1);
		for (int j = 0; j < N; j++){
			float value = cvGetReal2D(eigenface, j, 0);
				cvSetReal2D(colOfEigen, j, 0, value);
		}
		//cvNormalize(colOfEigen, colOfEigen, 255, 0, CV_MINMAX);
		//cvAddWeighted(colOfEigen, 0.1, add10, 1.0, 0, add10);
		cvAdd(add10, colOfEigen, add10);
	}
	//把add10映射到0-255
	cvNormalize(add10, add10, 255, 0, CV_MINMAX);
	// test value of add10
	for (int k = 0; k < N; k++){
		float value = cvGetReal2D(add10, k, 0);
		if (value>100)
			cout << value << "__";
	}
	//把add10 reshape成rows*cols
	CvMat* add10_reshape = cvCreateMat(rows, cols, CV_32FC1);
	CvMat add10_hdr;
	add10_reshape = cvReshape(add10, &add10_hdr, 1, rows);
	IplImage* add10_show = cvCreateImage(cvSize(cols, rows), IPL_DEPTH_8U, 1);
	IplImage add10_show_hdr;
	add10_show = cvGetImage(&add10_hdr, &add10_show_hdr);
	//cvShowImage("add10", add10_show);
	cvSaveImage("./output/add10lll.jpg", add10_show);
	*/
	/*
	IplImage* add10 = cvCreateImage(cvSize(cols, rows), IPL_DEPTH_16U, 1);
	for (int i = 0; i < 10; i++)
	{
		char file10[30];
		sprintf(file10, "./output/col%d.jpg", (i + 1));
		IplImage* col = cvLoadImage(file10);
		cvAdd(add10, col, add10);
	}
	cvNormalize(add10, add10, 255, 0, CV_MINMAX);
	cvSaveImage("./output/add10_2.bpm", add10);
	
	// pretend to show...
	IplImage* add10_test;
	add10_test = cvLoadImage("./output/add10_2.bpm");
	cvShowImage("add10", add10_test);
	*/
	cvWaitKey(0);

	cvReleaseImage(&Add10);

}
