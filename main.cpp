#include "detector.h"
#include "test.h"
#include <vector>


int main()
{
	std::vector<std::string> class_names = LoadNames("./weights/coco.names");   // 源文件为  ../weights/coco.names
	if (class_names.empty()) {
		return -1;
	}
	bool view_img = 1;
	torch::DeviceType device_type = torch::kCUDA;
	std::string weights = "./weights/best.torchscript.pt";
	auto detector = Detector(weights, device_type);

	cv::String path0 = "./images";
	std::vector<cv::String> files0;
	cv::glob(path0, files0, false);
	std::vector<cv::Mat> images0;
	for (int i = 0; i < files0.size(); i++)
	{
		cv::Mat src = cv::imread(files0[i]);
		images0.push_back(src);
	}

	infos info;
	std::vector<std::vector<infos>> result_infos;   //这里包含了每张图像中每个缺陷的检测信息
	float conf_thres = 0.50;  // 置信度阈值，域值越大，检测结果越少
	std::cout << "初始化开始..." << std::endl;
	detect1(images0, info, result_infos, conf_thres, class_names, detector, view_img);
	std::cout << "初始化完成, 开始检测" << std::endl;
	cv::String path = "E:\\wafer_test";
	std::vector<cv::String> files;
	cv::glob(path, files, false);

	std::vector<cv::Mat> images;
	for (int i = 0; i < files.size(); i++)
	{
		cv::Mat src = cv::imread(files[i]);
		images.push_back(src);
	}
	detect(images, info, result_infos, conf_thres, class_names, detector, view_img);

	std::cout << "result_infos.size: " << result_infos.size() << std::endl;
	int count = 0;
	for (int i = 0; i < result_infos.size(); i++)
	{
		if (result_infos[i].size() != 0)
		{
			std::cout << "\n=============" << count << "=============" << std::endl;
			for (int j = 0; j < result_infos[i].size(); j++)
			{
				std::cout << result_infos[i][j].location << std::endl;  // (左上角x，左上角y，宽，高)
				std::cout << result_infos[i][j].cls << std::endl;       // 类别
				std::cout << result_infos[i][j].conf << std::endl;      // 置信度
			}
			count += 1;
		}
		
	}

	return 0;
}