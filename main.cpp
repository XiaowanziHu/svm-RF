#include "stdafx.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "glcm.h"

using namespace cv;
using namespace cv::ml;
using namespace std;

Mat img, image;
Mat targetData1, targetData2, targetData3, targetData4;
bool flag = true;
int flag1 = 0;

void getTrainData4(Mat &train_data, Mat &train_label, string path1, string path2, string path3, string path4);  //生成训练数据 
void getTrainData3(Mat &train_data, Mat &train_label, string path2, string path3, string path4);  //生成训练数据 
void getTrainData5(Mat &train_data, Mat &train_label, string path1, string path2, string path3, string path4, string path5);
void TxtToMat(ifstream & infile, cv::Mat & Img);  //txt转Mat
void svm(); //svm分类
void colour(int height, int width, int* klabels, Mat imgRGB);//上色
void updateKlabels(int height, int width, int*klabels, string path);//更新标签
void m(int height, int width, int*klabels, string path1);
void Txttopointer(int height, int width, int *klabels, string path);//读取初始标签


int main()
{
	GLCM glcm;
	TextureEValues EValues;
	// 纹理特征值矩阵
	Mat imgEnergy, imgContrast, imgHomogenity, imgEntropy, imgData;
	string imgpath = "test-image\\yangben1.jpg";
	img = imread(imgpath);
	img.copyTo(image);

	int height = img.rows;
	int width = img.cols;
	int imgSize = height*width;
	int* klabels = nullptr;
	if (0 == klabels) klabels = new int[imgSize];
	int* klabels1 = nullptr;
	if (0 == klabels1) klabels1 = new int[3000];
	//svm();
	string inpath = "klabels1-3000";
	Txttopointer(height, width, klabels, inpath);
	//string Respath = "train-feature\\result\\1GLCM-30002.txt";
	//string Respath = "train-feature\\result\\1Spec-30002.txt";

	string Respath = "train-feature\\result\\1GLCM+Spec-30004.txt";
	//string Respath = "train-feature\\result\\1GLCM+Spec-30004-rdf+svm-3.txt";
	/*Txttopointer(height, width, klabels1, Respath1);

	ifstream infile1;
	infile1.open("F:\\hjm\\svm\\opencv-svm\\ConsoleApplication4\\train-klabel\\2\\1klabels-3000-daolu13", ios::binary | ios::app | ios::in | ios::out);
	string line;
	int i = 0;
	//标签融合
	while (getline(infile1, line))
	{
		istringstream stream(line);
		int x;
		while (stream >> x)
		{
			klabels1[x] = 4002;
		}
	}
	fstream out;
	out.open(Respath, ios::binary | ios::app | ios::in | ios::out);
	for (int i = 0; i < 3000; i++)
	{
		if (out.is_open())
		{
			out << klabels1[i] << "\r\n";
		}
	}
	out.close();*/

	updateKlabels(height, width, klabels, Respath);
	//像素与面向对象结合
	//string path1 = "pixel\\train-data-3\\label-pixel.txt";
	//m(height, width, klabels, path1);
	colour(height, width, klabels, img);

	if (klabels) delete[] klabels;
	if (klabels1) delete[] klabels1;
	imshow("image", img);
	//imwrite("2GLCM-result-4000-rbf-auto-test-feature.jpg", img);
	imwrite("1GLCM+Spec-result-3000-rbf-auto-test-feature4-12345.jpg", img);
	//imwrite("1GLCM+Spec-result-3000-rbf-auto-test-feature4+RDF-4.jpg", img);
	waitKey(0);
	getchar();
	return 0;
}

void svm()
{
	Mat train_data1, train_label1, train_data2, train_label2, train_data3, train_label3;
	GLCM glcm;
	Mat imgEnergy, imgContrast, imgHomogenity, imgEntropy, imgData;

	//getTrainData4(train_data2, train_label1, "train-feature\\featuresTraOut-gengdi-GLCM4.txt", "train-feature\\featuresTraOut-daolu-GLCM4.txt", "train-feature\\featuresTraOut-jianzhu-GLCM4.txt", "train-feature\\featuresTraOut-shuiti-GLCM4.txt");//效果好一点
	//getTrainData4(train_data3, train_label3, "train-feature\\featuresTraOut-gengdi-spec4.txt", "train-feature\\featuresTraOut-daolu-spec4.txt", "train-feature\\featuresTraOut-jianzhu-spec4.txt", "train-feature\\featuresTraOut-shuiti-spec4.txt");//效果好一点
	//getTrainData4(train_data1, train_label1, "train-feature\\1featuresTraOut-gengdi-GLCM_30002.txt", "train-feature\\1featuresTraOut-daolu-GLCM_30002.txt", "train-feature\\1featuresTraOut-jianzhu-GLCM_30002.txt", "train-feature\\1featuresTraOut-shuiti-GLCM_30002.txt");
	//getTrainData4(train_data1, train_label1, "train-feature\\1featuresTraOut-gengdi-Spec_30002.txt", "train-feature\\1featuresTraOut-daolu-Spec_30002.txt", "train-feature\\1featuresTraOut-jianzhu-Spec_30002.txt", "train-feature\\1featuresTraOut-shuiti-Spec_30002.txt");
	getTrainData4(train_data2, train_label1, "train-feature\\1\\1featuresTraOut-gengdi-GLCM_30004.txt", "train-feature\\1\\1featuresTraOut-daolu-GLCM_30004.txt", "train-feature\\1\\1featuresTraOut-jianzhu-GLCM_30004.txt", "train-feature\\1\\1featuresTraOut-shuiti-GLCM_30004.txt");
	getTrainData4(train_data3, train_label3, "train-feature\\1\\1featuresTraOut-gengdi-Spec_30004.txt", "train-feature\\1\\1featuresTraOut-daolu-Spec_30004.txt", "train-feature\\1\\1featuresTraOut-jianzhu-Spec_30004.txt", "train-feature\\1\\1featuresTraOut-shuiti-Spec_30004.txt");

	hconcat(train_data2, train_data3, train_data1);
	// 设置参数
	Ptr<SVM> svm1 = SVM::create();
	svm1->setType(SVM::C_SVC);

	//svm1->setKernel(SVM::LINEAR);

	svm1->setKernel(SVM::RBF);
	//svm1->setDegree(100);
	//svm1->setGamma(0.001);
	//svm1->setCoef0(0);
	//svm1->setC(10);
	//svm1->setNu(0.5);
	//svm1->setP(10);
	//svm1->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 2000, 0.0001));

	// 训练分类器
	//cout << train_data1 << endl;
	//cout << train_label1 << endl;

	Ptr<TrainData> tData1 = TrainData::create(train_data1, ROW_SAMPLE, train_label1);//这里的train_data是CV_32F；
	//svm1->train(tData1);

	svm1->trainAuto(tData1, 5,
	SVM::getDefaultGrid(SVM::C),
	SVM::getDefaultGrid(SVM::GAMMA),
	SVM::getDefaultGrid(SVM::P),
	SVM::getDefaultGrid(SVM::NU),
	SVM::getDefaultGrid(SVM::COEF),
	SVM::getDefaultGrid(SVM::DEGREE),
	false);

	cout << "train over" << endl;
	cout << "predict begin" << endl;

	//读取测试文件并转化为Mat类
	Mat test_data, test_dataGLCM, test_dataSpec;
	ifstream test_GLCM, test_Spec;
	test_GLCM.open("train-feature\\1featureFileGLCM-all-3000.txt", ios::binary | ios::app | ios::in | ios::out);
	test_Spec.open("train-feature\\1featureFileSpec-all-3000.txt", ios::binary | ios::app | ios::in | ios::out);
	TxtToMat(test_GLCM, test_dataGLCM);
	TxtToMat(test_Spec, test_dataSpec);
	hconcat(test_dataGLCM, test_dataSpec, test_data);
	cout << test_data.rows << endl;
	cout << test_data.cols << endl;	
	//cout << test_data<<endl;
	test_data.convertTo(test_data, CV_32F);//这里必须要转换数据格式
	for (int i = 0; i < test_data.rows; ++i)
	//for (int i = 0; i < train_data1.rows; ++i)
	{
		float *p_sample = test_data.ptr<float>(i);
		//float *p_sample = train_data1.ptr<float>(i);
		//cout << p_sample[0] << p_sample[1] << p_sample[2] << p_sample[3]<<endl;
		    Mat sampleMat(1, 8, CV_32FC1);
			sampleMat.at<float>(0, 0) = p_sample[0];
			sampleMat.at<float>(0, 1) = p_sample[1];
			sampleMat.at<float>(0, 2) = p_sample[2];
			sampleMat.at<float>(0, 3) = p_sample[3];
			sampleMat.at<float>(0, 4) = p_sample[4];
			sampleMat.at<float>(0, 5) = p_sample[5];
			sampleMat.at<float>(0, 6) = p_sample[6];
			sampleMat.at<float>(0, 7) = p_sample[7];
			//cout << sampleMat << endl;
			int response1 = svm1->predict(sampleMat);
			cout << response1 << endl;
			ofstream outfile;
			//outfile.open("train-feature\\Result\\1GLCM-30002.txt", ios::binary | ios::app | ios::in | ios::out);
			//outfile.open("train-feature\\Result\\1Spec-30002.txt", ios::binary | ios::app | ios::in | ios::out);
			outfile.open("train-feature\\Result\\1GLCM+Spec-30004.txt", ios::binary | ios::app | ios::in | ios::out);
			if (outfile.is_open())
			{
				outfile << (4000+response1);
				outfile << "\r\n";
			}
			outfile.close();
	}
	cout << "svm over" << endl;
}

void getTrainData3(Mat &train_data, Mat &train_label, string path2, string path3, string path4)
{
	Mat train_data1, train_data2, train_data3, train_data4, train_data5, train_data6;
	ifstream intfile1, intfile2, intfile3, intfile4;

	intfile2.open(path2, ios::binary | ios::app | ios::in | ios::out);
	TxtToMat(intfile2, train_data2);

	intfile3.open(path3, ios::binary | ios::app | ios::in | ios::out);
	TxtToMat(intfile3, train_data3);

	intfile4.open(path4, ios::binary | ios::app | ios::in | ios::out);
	TxtToMat(intfile4, train_data4);

	int m2 = train_data2.rows;
	int m3 = train_data3.rows;
	int m4 = train_data4.rows;


	vconcat(train_data2, train_data3, train_data5); //合并所有的样本点，作为训练数据
	vconcat(train_data5, train_data4, train_data);
	train_data.convertTo(train_data, CV_32F);
	//cout << train_data << endl;
	train_label = Mat(m2 + m3 + m4, 1, CV_32S, Scalar::all(1));//初始化标注

	for (int i = m2; i < (m2 + m3); i++)
		train_label.at<int>(i, 0) = 3;
	for (int i = (m2 + m3); i < (m2 + m3 + m4); i++)
		train_label.at<int>(i, 0) = 4;
	//cout << train_label << endl;
}

void getTrainData4(Mat &train_data, Mat &train_label, string path1, string path2, string path3, string path4)
{
	Mat train_data1, train_data2, train_data3, train_data4, train_data5, train_data6;
	ifstream intfile1, intfile2, intfile3, intfile4;
	intfile1.open(path1, ios::binary | ios::app | ios::in | ios::out);
	TxtToMat(intfile1, train_data1);

	intfile2.open(path2, ios::binary | ios::app | ios::in | ios::out);
	TxtToMat(intfile2, train_data2);

	intfile3.open(path3, ios::binary | ios::app | ios::in | ios::out);
	TxtToMat(intfile3, train_data3);

	intfile4.open(path4, ios::binary | ios::app | ios::in | ios::out);
	TxtToMat(intfile4, train_data4);

	int m1 = train_data1.rows;
	int m2 = train_data2.rows;
	int m3 = train_data3.rows;
	int m4 = train_data4.rows;
	vconcat(train_data1, train_data2, train_data5); //合并所有的样本点，作为训练数据
	vconcat(train_data5, train_data3, train_data6);
	vconcat(train_data6, train_data4, train_data);

	train_data.convertTo(train_data, CV_32F);
	//cout << train_data << endl;
	train_label = Mat((m1 + m2 + m3 + m4), 1, CV_32S, Scalar::all(1));//初始化标注

	for (int i = m1; i <(m1 + m2); i++)
	train_label.at<int>(i, 0) = 2;
	for (int i = (m1 + m2); i < (m1 + m2 + m3); i++)
		train_label.at<int>(i, 0) = 3;
	for (int i = (m1 + m2 + m3); i < (m1 + m2 + m3 + m4); i++)
		train_label.at<int>(i, 0) = 4;
	//cout << train_label << endl;
}

void getTrainData5(Mat &train_data, Mat &train_label, string path1, string path2, string path3, string path4,string path5)
{
	Mat train_data1, train_data2, train_data3, train_data4, train_data5, train_data6, train_data7, train_data8;
	ifstream intfile1, intfile2, intfile3, intfile4, intfile5;
	intfile1.open(path1, ios::binary | ios::app | ios::in | ios::out);
	TxtToMat(intfile1, train_data1);

	intfile2.open(path2, ios::binary | ios::app | ios::in | ios::out);
	TxtToMat(intfile2, train_data2);

	intfile3.open(path3, ios::binary | ios::app | ios::in | ios::out);
	TxtToMat(intfile3, train_data3);

	intfile4.open(path4, ios::binary | ios::app | ios::in | ios::out);
	TxtToMat(intfile4, train_data4); 

	intfile5.open(path5, ios::binary | ios::app | ios::in | ios::out);
	TxtToMat(intfile5, train_data5);

	int m1 = train_data1.rows;
	int m2 = train_data2.rows;
	int m3 = train_data3.rows;
	int m4 = train_data4.rows;
	int m5 = train_data5.rows;
	vconcat(train_data1, train_data2, train_data6); //合并所有的样本点，作为训练数据
	vconcat(train_data6, train_data3, train_data7);
	vconcat(train_data7, train_data4, train_data8);
	vconcat(train_data8, train_data5, train_data);

	train_data.convertTo(train_data, CV_32F);
	//cout << train_data << endl;
	train_label = Mat((m1 + m2 + m3 + m4 + m5), 1, CV_32S, Scalar::all(1));//初始化标注

	for (int i = m1; i <(m1 + m2); i++)
	train_label.at<int>(i, 0) = 2;
	for (int i = (m1 + m2); i < (m1 + m2 + m3); i++)
		train_label.at<int>(i, 0) = 3;
	for (int i = (m1 + m2 + m3); i < (m1 + m2 + m3 + m4); i++)
		train_label.at<int>(i, 0) = 4;
	for (int i = (m1 + m2 + m3 + m4); i < (m1 + m2 + m3 + m4 + m5); i++)
		train_label.at<int>(i, 0) = 5;
	//cout << train_label << endl;
}

void TxtToMat(ifstream & infile, cv::Mat & Img)
{
	string line;
	int num = 0;
	while (getline(infile, line)) {
		istringstream stream(line);
		double x;
		while (stream >> x) {
			Img.push_back(x);
		}
		num++;
	}
	Img = Img.reshape(1, num);
}

void Txttopointer(int height, int width, int *klabels,string path)
{
	string line;
	int num = 0;
	ifstream infile;
	infile.open(path, ios::binary | ios::app | ios::in | ios::out);
	while (getline(infile, line))
	{
		istringstream stream(line);
		double x;
		while (stream >> x)
		{
        	klabels[num] = x;
		}
		num++;
	}
	cout << "read over!" << endl;
}

void updateKlabels(int height, int width, int*klabels,string path)
{
	ifstream infile;
	infile.open(path, ios::binary | ios::app | ios::in | ios::out);
	//infile.open("response3", ios::binary | ios::app | ios::in | ios::out);
	string line;
	int num = 0;
	while (getline(infile, line))
	{
		istringstream stream(line);
		double x;
		while (stream >> x)
		{
			for (int j = 0; j < height; j++)
			{
				//获取第 i行首像素指针 
				for (int i = 0; i < width; i++)
				{
					int index = j*width + i;
					if (klabels[index] == num) klabels[index] = x;
				}
			}
		}
		num++;
	}
	cout << "updata over!" << endl;
}

void m(int height, int width, int*klabels, string path1)
{

	ifstream infile1;
	infile1.open(path1, ios::binary | ios::app | ios::in | ios::out);
	string line;
	int i = 0;
	int num = 0;
	//标签融合
	while (getline(infile1, line))
	{
		istringstream stream(line);
		double x;
		while (stream >> x)
		{
			int y = klabels[num];
			if (x == 2 && y == 4001) klabels[num] = 4002;
		}
		num++;
	}
	cout << "updata over!" << endl;
}

void colour(int height, int width, int* klabels, Mat imgRGB)
{
	Vec3b green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255), yellow(0, 255, 255), red2(0, 0, 0);
	for (int j = 0; j < height; j++)
	{
		//获取第 i行首像素指针 
		for (int i = 0; i < width; i++)
		{
			int index = j*width + i;
			int r = klabels[index];
			switch (r)
			{
			case 4001:
				imgRGB.at<Vec3b>(j, i) = green;//耕地
				break;
			case 4002:
				imgRGB.at<Vec3b>(j, i) = blue;//道路
				//imgRGB.at<Vec3b>(j, i) = green;//道路
				break;
			//case 4003:
				//imgRGB.at<Vec3b>(j, i) = red;//建筑
				//break;
			//case 4004:
				//imgRGB.at<Vec3b>(j, i) = yellow;//水体
				//break;
			case 4005:
				imgRGB.at<Vec3b>(j, i) = red2;//阴影
				break;
			}
		}
	}
}









