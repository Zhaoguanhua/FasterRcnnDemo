#pragma once
#define COMPILER_MSVC
#define NOMINMAX
#include <fstream>
#include <utility>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/env.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using namespace tensorflow;
using namespace tensorflow::ops;


// 定义FasterrcnnInterface类
class FasterrcnnInterface
{
public:
	/*
	---------------------变量---------------------
	*/
	FasterrcnnInterface(string model_path);
	/*
	---------------------方法---------------------
	*/
	//预测
	int predict(std::string input_img,std::string output_img);
	//边框微调使用tensor
	void bbox_transform_tensor(tensorflow::Tensor &rois, tensorflow::Tensor &delta_bbox, Eigen::Tensor<float, 2> &pred_boxes);
	//边框微调使用matrix
	void bbox_transform_matrix(tensorflow::Tensor &rois, tensorflow::Tensor &delta_bbox, 
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &pred_boxes, float im_scale);
	//向量排序
	void argsort(const Eigen::VectorXf& vec, Eigen::VectorXi& ind);
	//非极大抑制
	std::vector<int> cpu_nms(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&bboxes, Eigen::VectorXf &class_vec, float nms_thresh);
	//去除超出图片范围的边框
	void clip_bbox(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&pred_bboxes, 
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&bboxes, int tensor_height, int tensor_width);
	//返回满足条件的元素位置
	void select_where(Eigen::VectorXi &index_cond, Eigen::VectorXi &inds);

private:
	/*
	---------------------变量---------------------
	*/
	string inp_tensor_name_0;     // 输入节点名字(Placeholder)
	string inp_tensor_name_1;     // 输入节点名字(Placeholder_1)
	string out_tensor_name_bias;  // 输出节点名字(vgg_16_3/cls_score/BiasAdd)
	string out_tensor_name_score; // 输出节点名字(vgg_16_3/cls_prob)
	string out_tensor_name_bbox_pred;  // 输出节点名字(add)
	string out_tensor_name_rois;  // 输出节点名字(vgg_16_1/rois/concat)

	Session* session;  // 定义session
	GraphDef graphdef;  // 定义graph

	int tensor_max = 1000;// 定义图中的tensor尺寸
	int tensor_target = 600;//

	float CONF_THRESH = 0.8;
	float NMS_THRESH = 0.3;

	// 定义输入输出张量
	Tensor inp_Tensor_1 = Tensor(DT_FLOAT, TensorShape({ 3 }));  //输入张量

	std::vector<tensorflow::Tensor> out_Tensor;

	//classes
	std::vector<std::string> classes = {"background","aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair",
		"cow", "diningtable", "dog", "horse",
		"motorbike", "person", "pottedplant",
		"sheep", "sofa", "train", "tvmonitor" };

	//std::vector<std::string> classes = { "background","ship" };
	int n_classes = 21;

	//图像均值
	float r_mean = 122.7717;
	float g_mean = 115.9465;
	float b_mean = 102.9801;
};

