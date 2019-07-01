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
	svm();
	//RF();
	string inpath = "klabels1-3000";
	Txttopointer(height, width, klabels, inpath);
	string Respath = "train-feature\\result\\1GLCM+Spec-30004.txt";
        updateKlabels(height, width, klabels, Respath);
	colour(height, width, klabels, img);

	if (klabels) delete[] klabels;
	imshow("image", img);
	imwrite("2GLCM-result-4000-rbf-auto-test-feature.jpg", img);
	waitKey(0);
	return 0;
}

void svm()
{
	Mat train_data1, train_label1, train_data2, train_label2, train_data3, train_label3;
	GLCM glcm;
	getTrainData4(train_data2, train_label1, "train-feature\\1\\1featuresTraOut-gengdi-GLCM_30004.txt", "train-feature\\1\\1featuresTraOut-daolu-GLCM_30004.txt", "train-feature\\1\\1featuresTraOut-jianzhu-GLCM_30004.txt", "train-feature\\1\\1featuresTraOut-shuiti-GLCM_30004.txt");
	getTrainData4(train_data3, train_label3, "train-feature\\1\\1featuresTraOut-gengdi-Spec_30004.txt", "train-feature\\1\\1featuresTraOut-daolu-Spec_30004.txt", "train-feature\\1\\1featuresTraOut-jianzhu-Spec_30004.txt", "train-feature\\1\\1featuresTraOut-shuiti-Spec_30004.txt");

	hconcat(train_data2, train_data3, train_data1);
	// 设置参数
	Ptr<SVM> svm1 = SVM::create();
	svm1->setType(SVM::C_SVC);

	svm1->setKernel(SVM::RBF);
	Ptr<TrainData> tData1 = TrainData::create(train_data1, ROW_SAMPLE, train_label1);//这里的train_data是CV_32F；

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
		    Mat sampleMat(1, 8, CV_32FC1);
			sampleMat.at<float>(0, 0) = p_sample[0];
			sampleMat.at<float>(0, 1) = p_sample[1];
			sampleMat.at<float>(0, 2) = p_sample[2];
			sampleMat.at<float>(0, 3) = p_sample[3];
			sampleMat.at<float>(0, 4) = p_sample[4];
			sampleMat.at<float>(0, 5) = p_sample[5];
			sampleMat.at<float>(0, 6) = p_sample[6];
			sampleMat.at<float>(0, 7) = p_sample[7];
			int response1 = svm1->predict(sampleMat);
			cout << response1 << endl;
			ofstream outfile;
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
void RF()
{
	Mat train_data1, train_label1, train_data2, train_label2, train_data3, train_label3;
	GLCM glcm;
	getTrainData4(train_data2, train_label1, "train-feature\\1\\1featuresTraOut-gengdi-GLCM_30004.txt", "train-feature\\1\\1featuresTraOut-daolu-GLCM_30004.txt", "train-feature\\1\\1featuresTraOut-jianzhu-GLCM_30004.txt", "train-feature\\1\\1featuresTraOut-shuiti-GLCM_30004.txt");
	getTrainData4(train_data3, train_label3, "train-feature\\1\\1featuresTraOut-gengdi-Spec_30004.txt", "train-feature\\1\\1featuresTraOut-daolu-Spec_30004.txt", "train-feature\\1\\1featuresTraOut-jianzhu-Spec_30004.txt", "train-feature\\1\\1featuresTraOut-shuiti-Spec_30004.txt");

	hconcat(train_data2, train_data3, train_data1);

	int nsamples_all = train_data1.rows;  //样本总数
	int ntrain_samples = (int)(nsamples_all*0.8);  //训练样本个数
	cout << "Training the classifier ...\n" << endl;
	Mat sample_idx = Mat::zeros(1, train_data1.rows, CV_8U);
	int nvars = train_data1.cols;
	Mat var_type(nvars + 1, 1, CV_8U);
	var_type.setTo(Scalar::all(VAR_ORDERED));
	var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

	Ptr<TrainData> tData = TrainData::create(train_data1, ROW_SAMPLE, train_label1, noArray(), sample_idx, noArray(), var_type);//这里的train_data是CV_32F；

	// 创建分类器
	Ptr<RTrees> model;
	model = RTrees::create();
	//树的最大可能深度
	model->setMaxDepth(10);
	//节点最小样本数量
	model->setMinSampleCount(10);
	//回归树的终止标准
	model->setRegressionAccuracy(0);
	//是否建立替代分裂点
	model->setUseSurrogates(false);
	//最大聚类簇数
	model->setMaxCategories(15);
	//先验类概率数组
	model->setPriors(Mat());
	//计算的变量重要性
	model->setCalculateVarImportance(true);
	//树节点随机选择的特征子集的大小
	model->setActiveVarCount(4);
	//终止标准
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + (0.01f > 0 ? TermCriteria::EPS : 0), 100, 0.01f));
	//训练模型
	model->train(tData);
	//保存训练完成的模型
	//model->save("filename_to_save.xml");

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
			int response1 = model->predict(sampleMat);
			cout << response1 << endl;
			ofstream outfile;
			outfile.open("train-feature\\Result\\1GLCM+Spec-30004.txt", ios::binary | ios::app | ios::in | ios::out);
			if (outfile.is_open())
			{
				outfile << (4000+response1);
				outfile << "\r\n";
			}
			outfile.close();
	}
	//随机森林中的树个数
	cout << "Number of trees: " << model->getRoots().size() << endl;
	// 变量重要性
	Mat var_importance = model->getVarImportance();
	if (!var_importance.empty())
	{
		double rt_imp_sum = sum(var_importance)[0];
		printf("var#\timportance (in %%):\n");
		int i, n = (int)var_importance.total();
		for (i = 0; i < n; i++)
			printf("%-2d\t%-4.1f\n", i, 100.f*var_importance.at<float>(i) / rt_imp_sum);
	}
	cout << "over" << endl;
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
				break;
			case 4003:
				imgRGB.at<Vec3b>(j, i) = red;//建筑
				break;
			case 4004:
				imgRGB.at<Vec3b>(j, i) = yellow;//水体
				break;
			case 4005:
				imgRGB.at<Vec3b>(j, i) = red2;//阴影
				break;
			}
		}
	}
}









