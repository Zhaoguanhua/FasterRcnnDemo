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


// ����FasterrcnnInterface��
class FasterrcnnInterface
{
public:
	/*
	---------------------����---------------------
	*/
	FasterrcnnInterface(string model_path);
	/*
	---------------------����---------------------
	*/
	//Ԥ��
	int predict(std::string input_img,std::string output_img);
	//�߿�΢��ʹ��tensor
	void bbox_transform_tensor(tensorflow::Tensor &rois, tensorflow::Tensor &delta_bbox, Eigen::Tensor<float, 2> &pred_boxes);
	//�߿�΢��ʹ��matrix
	void bbox_transform_matrix(tensorflow::Tensor &rois, tensorflow::Tensor &delta_bbox, 
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &pred_boxes, float im_scale);
	//��������
	void argsort(const Eigen::VectorXf& vec, Eigen::VectorXi& ind);
	//�Ǽ�������
	std::vector<int> cpu_nms(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&bboxes, Eigen::VectorXf &class_vec, float nms_thresh);
	//ȥ������ͼƬ��Χ�ı߿�
	void clip_bbox(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&pred_bboxes, 
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&bboxes, int tensor_height, int tensor_width);
	//��������������Ԫ��λ��
	void select_where(Eigen::VectorXi &index_cond, Eigen::VectorXi &inds);

private:
	/*
	---------------------����---------------------
	*/
	string inp_tensor_name_0;     // ����ڵ�����(Placeholder)
	string inp_tensor_name_1;     // ����ڵ�����(Placeholder_1)
	string out_tensor_name_bias;  // ����ڵ�����(vgg_16_3/cls_score/BiasAdd)
	string out_tensor_name_score; // ����ڵ�����(vgg_16_3/cls_prob)
	string out_tensor_name_bbox_pred;  // ����ڵ�����(add)
	string out_tensor_name_rois;  // ����ڵ�����(vgg_16_1/rois/concat)

	Session* session;  // ����session
	GraphDef graphdef;  // ����graph

	int tensor_max = 1000;// ����ͼ�е�tensor�ߴ�
	int tensor_target = 600;//

	float CONF_THRESH = 0.8;
	float NMS_THRESH = 0.3;

	// ���������������
	Tensor inp_Tensor_1 = Tensor(DT_FLOAT, TensorShape({ 3 }));  //��������

	std::vector<tensorflow::Tensor> out_Tensor;

	//classes
	std::vector<std::string> classes = {"background","aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair",
		"cow", "diningtable", "dog", "horse",
		"motorbike", "person", "pottedplant",
		"sheep", "sofa", "train", "tvmonitor" };

	//std::vector<std::string> classes = { "background","ship" };
	int n_classes = 21;

	//ͼ���ֵ
	float r_mean = 122.7717;
	float g_mean = 115.9465;
	float b_mean = 102.9801;
};

