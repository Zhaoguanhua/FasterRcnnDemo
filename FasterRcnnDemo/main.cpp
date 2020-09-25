#include <iostream>
#include <vector>
#include "faster_rcnn_interface.h"

int main(int argc,char **argv)
{

	const char *img_path = argv[1];
	const char *out_path = argv[2];
	const char *mode_path = argv[3];

	FasterrcnnInterface fasterrcnn = FasterrcnnInterface(mode_path);
	//Eigen::MatrixXf objects;
	fasterrcnn.predict(img_path,out_path);

	return 0;



}