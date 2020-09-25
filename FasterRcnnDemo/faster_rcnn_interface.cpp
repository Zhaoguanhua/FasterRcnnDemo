#include<cmath>
#include<ctime>
#include "faster_rcnn_interface.h"
#include<typeinfo>
#include<sstream>
#include<iomanip>

FasterrcnnInterface::FasterrcnnInterface(string model_path)
{
	//// 配置模型的输入输出节点名字 vgg16
	//inp_tensor_name_0 = "Placeholder:0";
	//inp_tensor_name_1 = "Placeholder_1:0";
	//out_tensor_name_bias = "vgg_16_3/cls_score/BiasAdd:0";
	//out_tensor_name_score = "vgg_16_3/cls_prob:0";
	//out_tensor_name_bbox_pred = "add:0";
	//out_tensor_name_rois = "vgg_16_1/rois/concat:0";

	// 配置模型的输入输出节点名字 res101
	inp_tensor_name_0 = "Placeholder:0";
	inp_tensor_name_1 = "Placeholder_1:0";
	out_tensor_name_bias = "resnet_v1_101_5/cls_score/BiasAdd:0";
	out_tensor_name_score = "resnet_v1_101_5/cls_prob:0";
	out_tensor_name_bbox_pred = "add:0";
	out_tensor_name_rois = "resnet_v1_101_3/rois/concat:0";


	// 加载模型到计算图
	Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef);
	if (!status_load.ok()) {
		std::cout << "ERROR: Loading model failed..." << std::endl;
		std::cout << model_path << status_load.ToString() << "\n";
		exit(0);
	}

	// 创建会话
	NewSession(SessionOptions(), &session);
	Status status_create = session->Create(graphdef);
	if (!status_create.ok()) {
		std::cout << "ERROR: Creating graph in session failed.." << status_create.ToString() << std::endl;
		exit(0);
	}
	else {
		std::cout << "----------- Successfully created session and load graph -------------" << std::endl;
	}

}

int FasterrcnnInterface::predict(std::string input_img,std::string output_img)
{

	//读取数据
	cv::Mat srcImage = cv::imread(input_img, 1);

	if (srcImage.empty()) {  // 校验是否正常打开待操作图像!
		std::cout << "can't open the image!!!!!!!" << std::endl;
	}

	int img_height = srcImage.rows;  //图片高
	int img_width = srcImage.cols;   //图片宽

	cv::Mat inpFloatMat;

	srcImage.copyTo(inpFloatMat);    //转换类型
	inpFloatMat.convertTo(inpFloatMat, CV_32FC3);

	//去中心化
	cv::subtract(inpFloatMat, cv::Scalar(b_mean,g_mean,r_mean), inpFloatMat);

	float img_size_min,img_size_max;

	//判断图片尺寸的最大值和最小值
	if (img_height>=img_width)
	{
		img_size_min = img_width;
		img_size_max = img_height;
	}
	else
	{
		img_size_min = img_height;
		img_size_max = img_width;
	}

	//先按照目标尺寸和图片尺寸最小值进行缩放比计算
	float im_scale = tensor_target / img_size_min;

	//如果缩放后的图片的最大值仍然比设置的tensor尺寸的最大值还大，需要重新按照最大值来进行
	//缩放系数的计算
	if (im_scale*img_size_max>tensor_max)
	{
		im_scale = tensor_max / img_size_max;
	}

	int tensor_height = int(img_height*im_scale);
	int tensor_width = int(img_width*im_scale);

	//重采样
	cv::Mat resize_img;

	cv::resize(inpFloatMat, resize_img, cv::Size(tensor_width, tensor_height));

	//输入tensort inp_Tensor_0填充数据
	Tensor inp_Tensor_0 = Tensor(DT_FLOAT, TensorShape({ 1, tensor_height, tensor_width, 3 }));  // 输入张量
	float *tensor_data_ptr = inp_Tensor_0.flat<float>().data();
	cv::Mat fake_mat(tensor_height, tensor_width, CV_32FC(3), tensor_data_ptr);
	resize_img.convertTo(fake_mat, CV_32FC(3));

	//输入tensort inp_Tensor_1填充数据
	auto tmap_info = inp_Tensor_1.tensor<float, 1>();

	tmap_info(0) = tensor_height;
	tmap_info(1) = tensor_width;
	tmap_info(2) = im_scale;


	std::vector<std::pair<string, Tensor>> inputs = {
		{ inp_tensor_name_0,inp_Tensor_0 },
		{ inp_tensor_name_1,inp_Tensor_1 }
	};

	std::vector<string> output_tensor_names = { out_tensor_name_bias,out_tensor_name_score,
		out_tensor_name_bbox_pred,out_tensor_name_rois };

	// 输入张量 -> 输出张量
	Status status_run = session->Run({ inputs }, { output_tensor_names }, {}, &out_Tensor);
	if (!status_run.ok()) {
		std::cout << "ERROR: RUN failed..." << std::endl;
		std::cout << status_run.ToString() << "\n";
		exit(0);
	}

	//边框修正
	//输出Tensor中，Tensor[1]是得分，Tensor[2]是检测框修正值,Tensor[3]是检测框
	Eigen::MatrixXf pred_boxes(300, n_classes*4);
	//Eigen::MatrixXf pred_boxes(300, 8);
	bbox_transform_matrix(out_Tensor[3], out_Tensor[2], pred_boxes,im_scale);

	//裁剪超过边界的边框
	Eigen::MatrixXf bboxes(300, n_classes*4);
	//Eigen::MatrixXf bboxes(300, 8);
	clip_bbox(pred_boxes, bboxes,tensor_height,tensor_width);
	
	auto m_scores = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic,
		Eigen::Dynamic, Eigen::RowMajor>>(out_Tensor[1].flat<float>().data(), 300, n_classes);

	for (int class_id=1;class_id<n_classes;class_id++)
	{
		//非极大抑制NMS
		std::vector<int> keep;
		Eigen::VectorXf class_vec = m_scores.col(class_id);

		//Eigen::MatrixXf bboxes_id = bboxes.block(0, 4 * class_id, 4, 300);
		Eigen::MatrixXf bboxes_id(300,4);
		bboxes_id.col(0) = bboxes.col(4*class_id);
		bboxes_id.col(1) = bboxes.col(4 * class_id+1);
		bboxes_id.col(2) = bboxes.col(4 * class_id+2);
		bboxes_id.col(3) = bboxes.col(4 * class_id+3);
		keep = cpu_nms(bboxes_id, class_vec, NMS_THRESH);
		int keep_size = keep.size();
		auto keep_matrix = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(keep.data(), 1, keep_size);
		//std::cout << "keep" << keep_matrix << std::endl;

		//选出得分超过阈值的边框
		Eigen::VectorXi keep_vec = keep_matrix.row(0);
		Eigen::MatrixXf dets(keep_size, 4);
		dets.col(0) = bboxes_id.col(0)(keep_vec);
		dets.col(1) = bboxes_id.col(1)(keep_vec);
		dets.col(2) = bboxes_id.col(2)(keep_vec);
		dets.col(3) = bboxes_id.col(3)(keep_vec);
		//auto m_scores = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(out_Tensor[1].flat<float>().data(), 300, 2);

		Eigen::VectorXf dets_score = class_vec(keep_vec);

		Eigen::VectorXi score_cond = (dets_score.array() >= CONF_THRESH).cast<int>();
		//std::cout << "index_cond:" << index_cond.transpose() << std::endl;
		int cond_sum = score_cond.sum();

		Eigen::VectorXi inds(cond_sum);
		inds.setZero();
		select_where(score_cond, inds);

		for (int id = 0; id < inds.size(); id++)
		{
			cv::Point pt1, pt2;
			pt1.x = dets.row(inds[id])[0];
			pt1.y = dets.row(inds[id])[1];
			pt2.x = dets.row(inds[id])[2];
			pt2.y = dets.row(inds[id])[3];

			//std::string score_class = std::to_string(dets_score[inds[id]]);

			std::stringstream text;
			//std::cout << std::setprecision(3);
			text << std::setprecision(3)<< classes[class_id] << ":" << dets_score[inds[id]];

			cv::rectangle(srcImage, pt1, pt2, cv::Scalar(0, 0, 255), 1, 1, 0);
			cv::putText(srcImage, text.str(), pt1, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1, CV_AA);
		}
	}

	cv::imwrite(output_img, srcImage);

	return(0);
}

void FasterrcnnInterface::bbox_transform_tensor(tensorflow::Tensor &rois, tensorflow::Tensor &delta_bbox, Eigen::Tensor<float, 2> &pred_boxes)
{
	//tensorflow::tensor转engin::tensor
	Eigen::Tensor<float, 2> t_rois = Eigen::TensorLayoutSwapOp<Eigen::Tensor<float, 2, Eigen::RowMajor>>(rois.tensor<float, 2>());
	Eigen::Tensor< float, 2> t_delta_bbox = Eigen::TensorLayoutSwapOp<Eigen::Tensor<float, 2, Eigen::RowMajor>>(delta_bbox.tensor<float, 2>());

	Eigen::Tensor<float, 1> rois_x1 = t_rois.chip(1, 0);
	Eigen::Tensor<float, 1> rois_y1 = t_rois.chip(2, 0);
	Eigen::Tensor<float, 1> rois_x2 = t_rois.chip(3, 0);
	Eigen::Tensor<float, 1> rois_y2 = t_rois.chip(4, 0);

	Eigen::Tensor<float, 1> addis(300);
	addis.setConstant(1.0f);

	//(x1,y1,x2,y2)转换成(center_x,center_y,heights,widths)
	Eigen::Tensor<float, 1>	widths = rois_x2 - rois_x1 + addis;
	Eigen::Tensor<float, 1>	heights = rois_y2 - rois_y1 + addis;
	Eigen::Tensor<float, 1>	ctr_x = rois_x1 + 0.5*widths;
	Eigen::Tensor<float, 1>	ctr_y = rois_y1 + 0.5*heights;

	//中心点x坐标偏移
	Eigen::Tensor<float, 2> dx(2, 300);
	dx.setZero();
	dx.chip(0, 0) = t_delta_bbox.chip(0, 0);
	dx.chip(1, 0) = t_delta_bbox.chip(4, 0);
	//中心点y坐标偏移
	Eigen::Tensor<float, 2> dy(2, 300);
	dy.setZero();
	dy.chip(0, 0) = t_delta_bbox.chip(1, 0);
	dy.chip(1, 0) = t_delta_bbox.chip(5, 0);
	//bounding box宽缩放系数
	Eigen::Tensor<float, 2> dw(2, 300);
	dw.setZero();
	dw.chip(0, 0) = t_delta_bbox.chip(2, 0);
	dw.chip(1, 0) = t_delta_bbox.chip(6, 0);
	//bounding box高缩放系数
	Eigen::Tensor<float, 2> dh(2, 300);
	dh.setZero();
	dh.chip(0, 0) = t_delta_bbox.chip(3, 0);
	dh.chip(1, 0) = t_delta_bbox.chip(7, 0);

	Eigen::Tensor<float, 2>pred_ctr_x(2, 300);
	pred_ctr_x.chip(0, 0) = dx.chip(0, 0)*widths + ctr_x;
	pred_ctr_x.chip(1, 0) = dx.chip(1, 0)*widths + ctr_x;

	Eigen::Tensor<float, 2>pred_ctr_y(2, 300);
	pred_ctr_y.chip(0, 0) = dy.chip(0, 0)*heights + ctr_y;
	pred_ctr_y.chip(1, 0) = dy.chip(1, 0)*heights + ctr_y;

	Eigen::Tensor<float, 2>pred_w(2, 300);
	pred_w.chip(0, 0) = dw.chip(0, 0).exp()*widths;
	pred_w.chip(1, 0) = dw.chip(1, 0).exp()*widths;

	Eigen::Tensor<float, 2>pred_h(2, 300);
	pred_h.chip(0, 0) = dh.chip(0, 0).exp()*heights;
	pred_h.chip(1, 0) = dh.chip(1, 0).exp()*heights;

	//(center_x,center_y,height,width) 转换成(x1,y1,x2,y2)
	pred_boxes.chip(0, 0) = pred_ctr_x.chip(0, 0) - 0.5*pred_w.chip(0, 0);
	pred_boxes.chip(4, 0) = pred_ctr_x.chip(1, 0) - 0.5*pred_w.chip(1, 0);

	pred_boxes.chip(1, 0) = pred_ctr_y.chip(0, 0) - 0.5*pred_h.chip(0, 0);
	pred_boxes.chip(5, 0) = pred_ctr_y.chip(1, 0) - 0.5*pred_h.chip(1, 0);

	pred_boxes.chip(2, 0) = pred_ctr_x.chip(0, 0) + 0.5*pred_w.chip(0, 0);
	pred_boxes.chip(6, 0) = pred_ctr_x.chip(1, 0) + 0.5*pred_w.chip(1, 0);

	pred_boxes.chip(3, 0) = pred_ctr_y.chip(0, 0) + 0.5*pred_h.chip(0, 0);
	pred_boxes.chip(7, 0) = pred_ctr_y.chip(1, 0) + 0.5*pred_h.chip(1, 0);



}


void FasterrcnnInterface::bbox_transform_matrix(tensorflow::Tensor &rois, 
	tensorflow::Tensor &delta_bbox, Eigen::Matrix<float, Eigen::Dynamic, 
	Eigen::Dynamic> &pred_boxes,float im_scale)
{
	//Tensorflow::Tensor转换为Eigen::Matrix
	auto m_rois = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(rois.flat<float>().data(), 300, 5);
	auto m_delta_bbox = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(delta_bbox.flat<float>().data(), 300, n_classes*4);

	Eigen::VectorXf rois_x1 = m_rois.col(1) / im_scale;
	Eigen::VectorXf rois_y1 = m_rois.col(2) / im_scale;
	Eigen::VectorXf rois_x2 = m_rois.col(3) / im_scale;
	Eigen::VectorXf rois_y2 = m_rois.col(4) / im_scale;

	Eigen::VectorXf offset(300);
	offset.setConstant(1.0);

	//(x1,y1,x2,y2)转换成(center_x,center_y,heights,widths)
	Eigen::VectorXf	widths = rois_x2 - rois_x1 + offset;
	Eigen::VectorXf	heights = rois_y2 - rois_y1 + offset;
	Eigen::VectorXf	ctr_x = rois_x1 + 0.5*widths;
	Eigen::VectorXf	ctr_y = rois_y1 + 0.5*heights;
	
	Eigen::MatrixXf dx(300, n_classes);
	Eigen::MatrixXf dy(300, n_classes);
	Eigen::MatrixXf dw(300, n_classes);
	Eigen::MatrixXf dh(300, n_classes);


	for (int i=0;i<n_classes;i++)
	{
		//中心点x坐标偏移
		dx.col(i) = m_delta_bbox.col(0+4*i);

		//中心点y坐标偏移
		dy.col(i) = m_delta_bbox.col(1+4*i);

		//bounding box宽缩放系数
		dw.col(i) = m_delta_bbox.col(2+4*i);

		//bounding box高缩放系数
		dh.col(i) = m_delta_bbox.col(3+4*i);
	}

	////计算修改后的bounding box
	Eigen::MatrixXf pred_ctr_x(300, n_classes);
	Eigen::MatrixXf pred_ctr_y(300, n_classes);
	Eigen::MatrixXf pred_w(300, n_classes);
	Eigen::MatrixXf pred_h(300, n_classes);


	for (int i = 0; i < n_classes; i++)
	{
		
		pred_ctr_x.col(i) = dx.col(i).cwiseProduct(widths) + ctr_x;

		pred_ctr_y.col(i) = dy.col(i).cwiseProduct(heights) + ctr_y;

		pred_w.col(i) = Eigen::VectorXf(dw.col(i).array().exp()).cwiseProduct(widths);

		pred_h.col(i) = Eigen::VectorXf(dh.col(i).array().exp()).cwiseProduct(heights);
	}

	////修正边框(center_x,center_y,h,w)转(x1,y1,x2,y2)
	for (int i=0;i<n_classes;i++)
	{
		
		pred_boxes.col(0+4*i) = pred_ctr_x.col(i) - 0.5*pred_w.col(i);
		//pred_boxes.col(4) = pred_ctr_x.col(1) - 0.5*pred_w.col(1);

		pred_boxes.col(1+4*i) = pred_ctr_y.col(i) - 0.5*pred_h.col(i);
		//pred_boxes.col(5) = pred_ctr_y.col(1) - 0.5*pred_h.col(1);

		pred_boxes.col(2+4*i) = pred_ctr_x.col(i) + 0.5*pred_w.col(i);
		//pred_boxes.col(6) = pred_ctr_x.col(1) + 0.5*pred_w.col(1);

		pred_boxes.col(3+4*i) = pred_ctr_y.col(i) + 0.5*pred_h.col(i);
		//pred_boxes.col(7) = pred_ctr_y.col(1) + 0.5*pred_h.col(1);
	}

}

void FasterrcnnInterface::argsort(const Eigen::VectorXf& vec, Eigen::VectorXi& ind)
{
	ind = Eigen::VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1);//[0 1 2 3 ... N-1]
	auto rule = [vec](int i, int j)->bool {
		return vec(i) > vec(j);
	};
	std::sort(ind.data(), ind.data() + ind.size(), rule);
}

std::vector<int> FasterrcnnInterface::cpu_nms(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&bboxes, Eigen::VectorXf &class_vec, float nms_thresh)
{

	//按照得分排序
	Eigen::VectorXi score_order(300);
	argsort(class_vec, score_order);

	Eigen::VectorXf x1 = bboxes.col(0);
	Eigen::VectorXf y1 = bboxes.col(1);
	Eigen::VectorXf x2 = bboxes.col(2);
	Eigen::VectorXf y2 = bboxes.col(3);

	Eigen::VectorXf offset(300);
	offset.setConstant(1.0);

	std::vector<int> keep;

	//计算bounding box面积
	Eigen::VectorXf areas = (x2 - x1 + offset).cwiseProduct(y2 - y1 + offset);

	while (score_order.size() > 0)
	{
		//std::cout << "order:" << score_order.transpose() << std::endl;
		int i = score_order[0];
		keep.push_back(i);
		//std::cout <<"keep:"<< i << std::endl;
		int order_size = score_order.size();
		Eigen::VectorXf xx1 = x1(score_order.segment(1, order_size - 1)).cwiseMax(x1[i]);
		Eigen::VectorXf yy1 = y1(score_order.segment(1, order_size - 1)).cwiseMax(y1[i]);
		Eigen::VectorXf xx2 = x2(score_order.segment(1, order_size - 1)).cwiseMin(x2[i]);
		Eigen::VectorXf yy2 = y2(score_order.segment(1, order_size - 1)).cwiseMin(y2[i]);

		Eigen::VectorXf w = (xx2 - xx1 + Eigen::VectorXf::Ones(order_size - 1)).cwiseMax(0);
		Eigen::VectorXf h = (yy2 - yy1 + Eigen::VectorXf::Ones(order_size - 1)).cwiseMax(0);
		Eigen::VectorXf inter = w.cwiseProduct(h);

		Eigen::VectorXf area_score_max(order_size - 1);
		area_score_max.setConstant(areas[i]);

		Eigen::VectorXf ovr = inter.cwiseQuotient(area_score_max + areas(score_order.segment(1, order_size - 1)) - inter);

		Eigen::VectorXi index_cond = (ovr.array() < nms_thresh).cast<int>();
		//std::cout << "index_cond:" << index_cond.transpose() << std::endl;
		int cond_sum = index_cond.sum();

		Eigen::VectorXi inds(cond_sum);
		inds.setZero();
		select_where(index_cond, inds);
		//std::cout << "inds:" << inds.transpose() << std::endl;
		Eigen::VectorXi offset2(cond_sum);
		offset2.setConstant(1);

		Eigen::VectorXi selcet_elem = score_order(inds + offset2);
		score_order = selcet_elem;
		//std::cout << "selcet_elem:" << selcet_elem.transpose() << std::endl;
	}
	return keep;
}


void FasterrcnnInterface::clip_bbox(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&pred_bboxes,
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&bboxes,int tensor_height,int tensor_width)
{
	for (int i=0;i<n_classes;i++)
	{
		////x1>=0
		bboxes.col(0+4*i) = pred_bboxes.col(0 + 4 * i).cwiseMax(0);

		////y1>=0
		bboxes.col(1 + 4 * i) = pred_bboxes.col(1 + 4 * i).cwiseMax(0);

		////x2<img_cols
		bboxes.col(2 + 4 * i) = pred_bboxes.col(2 + 4 * i).cwiseMin(tensor_width - 1);

		////y2<img_rows
		bboxes.col(3 + 4 * i) = pred_bboxes.col(3 + 4 * i).cwiseMin(tensor_height - 1);
	}

}

void FasterrcnnInterface::select_where(Eigen::VectorXi &index_cond, Eigen::VectorXi &inds)
{

	int j = 0;
	int vec_size = index_cond.size();
	for (int i = 0; i < vec_size; i++)
	{
		if (index_cond[i] == 1)
		{
			inds[j] = i;
			j++;
		}
	}
}

