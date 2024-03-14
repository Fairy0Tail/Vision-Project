#pragma once
#include <iostream>
#include <memory>
#include <chrono>
#include "cxxopts.hpp"


struct infos
{
    std::vector<int> location;
    std::string cls;
    std::string conf;
};



// new_add
static int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names) {
    DIR* p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}


std::vector<std::string> LoadNames(const std::string& path) {
    // load class names
    std::vector<std::string> class_names;
    std::ifstream infile(path);
    if (infile.is_open()) {
        std::string line;
        while (getline(infile, line)) {
            class_names.emplace_back(line);
        }
        infile.close();
    }
    else {
        std::cerr << "Error loading the class names!\n";
    }

    return class_names;
}


std::vector<cv::Rect> Demo(cv::Mat& img,
    const std::vector<std::vector<Detection>>& detections,
    const std::vector<std::string>& class_names,
    std::vector<std::string>& cls,
    std::vector<std::string>& conf,
    bool label = true)
{
    if (!detections.empty()) 
    {
        std::vector<cv::Rect> boxes;
        for (const auto& detection : detections[0]) 
        {
            const auto& box = detection.bbox;
            float score = detection.score;
            int class_idx = detection.class_idx;


            if (label)
            {
                cv::rectangle(img, box, cv::Scalar(0, 0, 255), 1);

                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << score;
                std::string s = class_names[class_idx] + " " + ss.str();

                cls.push_back(class_names[class_idx]);
                conf.push_back(ss.str());

                std::cout << "class: " << s << std::endl;

                auto font_face = cv::FONT_HERSHEY_DUPLEX;
                auto font_scale = 1.0;
                int thickness = 1;
                int baseline = 0;
                auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
                cv::rectangle(img,
                    cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                    cv::Point(box.tl().x + s_size.width, box.tl().y),
                    cv::Scalar(0, 0, 255), -1);
                cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
                    font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
            }
            boxes.push_back(box);
        }
        return boxes;
    }
    else
    {
        std::vector<cv::Rect> ret;
        ret.push_back(cv::Rect(0, 0, 0, 0));
        return ret;
    }
}


int detect(std::vector<cv::Mat> images,
           infos& info,
           std::vector< std::vector<infos>>& result_infos,
           float conf_thres,
           std::vector<std::string> class_names,
           Detector detector,
           bool view_img)
{
    //----------------------------------------------
    //bool view_img = "true";

    //// set device type - CPU/GPU
    //torch::DeviceType device_type = torch::kCUDA;;

    //// load class names from dataset for visualization
    //std::vector<std::string> class_names = LoadNames("./weights/coco.names");   // 源文件为  ../weights/coco.names
    //if (class_names.empty()) {
    //    return -1;
    //}
   
    //// load network
    //std::string weights = "./weights/best.torchscript.pt";
    //auto detector = Detector(weights, device_type);

    //----------------------------------------------

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < images.size(); i += 1) {
        //Get a batch of images
        //std::vector<cv::Mat> img_batch;
        //std::vector<std::string> img_name_batch;
        std::vector<infos> result_info;
        for (size_t j = i; j < i + 1 && j < images.size(); j++)
        {
            cv::Mat img = images[j];
            if (img.empty()) {
                std::cerr << "Error loading the image!\n";
                return -1;
            }


            // set up threshold
            float conf_thres = 0.4;
            float iou_thres = 0.5;

            std::cout << "----------------------------------------------------" << std::endl;

            std::vector<cv::Rect> box;
            std::vector<cv::Rect> boxes;
            std::vector<std::string> cls;
            std::vector<std::string> conf;
            //cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
            auto result = detector.Run(img, conf_thres, iou_thres);
            boxes = Demo(img, result, class_names, cls, conf);

            std::cout << "boxes size: " << boxes.size() << std::endl;

            if (boxes.size() != 0)
            {
                if (boxes[0] == cv::Rect(0, 0, 0, 0))
                {
                    box.push_back(boxes[0]);
                }
                else
                {
                    for (int k = 0; k < boxes.size(); k++)
                    {
                        box.push_back(boxes[k]);
                    }
                }
            }

            std::vector<int> flag;
            for (int i = 0; i < box.size(); i++)
            {
                if (box[i] == cv::Rect(0, 0, 0, 0))
                {
                    flag.push_back(0);
                }
                else
                {
                    flag.push_back(1);
                }
            }
            
            bool allZero = true;
            for (int i = 0; i < flag.size(); i++)
            {
                if (flag[i] != 0)
                {
                    allZero = false;
                    break;
                }
            }

            std::vector<cv::Rect> box_list;

            if (allZero)
            {
                std::vector<int> location;

                location.push_back(0);
                location.push_back(0);
                location.push_back(0);
                location.push_back(0);

                info.location = location;
                info.cls = "None";
                info.conf = "None";
                result_info.push_back(info);
            }
            else
            {
                for (int k = 0; k < flag.size(); k++)
                {
                    if (flag[k] != 0)
                    {
                        box_list.push_back(box[k]);
                    }
                }
            }

            std::cout << "box_list.size: " << box_list.size() << std::endl;
            std::cout << "cls.size: " << cls.size() << std::endl;
            std::cout << "conf.size: " << conf.size() << std::endl;

            for (int i = 0; i < box_list.size(); i++)
            {

                std::vector<int> location;
 
                location.push_back(box_list[i].x);
                location.push_back(box_list[i].y);
                location.push_back(box_list[i].width);
                location.push_back(box_list[i].height);
                info.location = location;
                info.cls = cls[i];
                info.conf = conf[i];

                result_info.push_back(info);
            }
            
            //if (boxes.size() != 0)
            //{
            //    if (boxes[0] != cv::Rect(0, 0, 0, 0))
            //    {
            //        for (int i = 0; i < boxes.size(); i++)
            //        {

            //            int center_x = boxes[i].x + boxes[i].width / 2 + 1;
            //            int center_y = boxes[i].y + boxes[i].height / 2 + 1;

            //            info.location = cv::Point(center_x, center_y);
            //            info.cls = cls[i];
            //            info.conf = conf[i];

            //            result_info.push_back(info);
            //        }


            //    }
            //}
            //std::string route = "C:\\Users\\Lenovo\\Desktop\\9\\";
            //cv::String filename = route + "dst" + std::to_string(i) + ".png";
            //cv::imwrite(filename, img);
            //std::cout << "----------------------------------------------------" << std::endl;
        }
        result_infos.push_back(result_info);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // It should be known that it takes longer time at first time
    std::cout << "Total time consumed : " << duration.count() / 1000.0 << " s" << std::endl;
}

int detect1(std::vector<cv::Mat> images,
    infos& info,
    std::vector< std::vector<infos>>& result_infos,
    float conf_thres,
    std::vector<std::string> class_names,
    Detector detector,
    bool view_img)
{

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < images.size(); i += 1) {
        // Get a batch of images
        //std::vector<cv::Mat> img_batch;
        //std::vector<std::string> img_name_batch;
        std::vector<infos> result_info;
        for (size_t j = i; j < i + 1 && j < images.size(); j++)
        {
            cv::Mat img = images[j];
            if (img.empty()) {
                std::cerr << "Error loading the image!\n";
                return -1;
            }

            float iou_thres = 0.5;
            //cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
            auto result = detector.Run1(img, conf_thres, iou_thres);
 
        }
    }
}
