#include "linemodLevelup.h"
#include <memory>
#include <iostream>
#include "linemod_icp.h"
#include <assert.h>
#include <chrono>  // for high_resolution_clock
#include <opencv2/rgbd.hpp>
using namespace std;
using namespace cv;

static std::string prefix = "/home/meiqua/6DPose/linemodLevelup/test/case1/";
// for test
std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
void train_test(){
    Mat rgb = cv::imread(prefix+"train_rgb.png");
    Mat depth = cv::imread(prefix+"train_dep.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    Mat mask = cv::imread(prefix+"train_mask.png");
    cvtColor(mask, mask, cv::COLOR_RGB2GRAY);
    cout << "rgb: " << type2str(rgb.type())  << endl;
    cout << "depth: " << type2str(depth.type())  << endl;
    cout << "mask: " << type2str(mask.type())  << endl;
    vector<Mat> sources;
    sources.push_back(rgb);
    sources.push_back(depth);
    auto detector = linemodLevelup::Detector();
    detector.addTemplate(sources, "06_template", mask);
    detector.writeClasses(prefix+"writeClasses/%s.yaml");
    cout << "break point line: train_test" << endl;
}

int main(){
    // test case1
    /*
     * (x=327, y=127, float similarity=92.66, class_id=06_template, template_id=424)
     * render K R t:
  cam_K: [572.41140000, 0.00000000, 325.26110000, 0.00000000, 573.57043000, 242.04899000, 0.00000000, 0.00000000, 1.00000000]
  cam_R_w2c: [0.34768538, 0.93761126, 0.00000000, 0.70540612, -0.26157897, -0.65877056, -0.61767070, 0.22904489, -0.75234390]
  cam_t_w2c: [0.00000000, 0.00000000, 1000.00000000]

  gt K R t:
- cam_R_m2c: [0.09506610, 0.98330897, -0.15512900, 0.74159598, -0.17391300, -0.64791101, -0.66407597, -0.05344890, -0.74575198]
  cam_t_m2c: [71.62781422, -158.20064191, 1050.77777823]
  obj_bb: [331, 130, 65, 64]
  obj_id: 6
  */

//    train_test();

    Mat rgb = cv::imread(prefix+"0000_rgb.png");
    Mat depth = cv::imread(prefix+"0000_dep.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    vector<Mat> sources;
    sources.push_back(rgb);
    sources.push_back(depth);
//    cout << "depth: " << type2str(depth.type())  << endl;

    Mat K_ren = (Mat_<float>(3,3) << 572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0);
    Mat R_ren = (Mat_<float>(3,3) << 0.34768538, 0.93761126, 0.00000000, 0.70540612,
                 -0.26157897, -0.65877056, -0.61767070, 0.22904489, -0.75234390);
    Mat t_ren = (Mat_<float>(3,1) << 0.0, 0.0, 1000.0);

    auto detector = linemodLevelup::Detector();
    auto ori_detector = cv::linemod::getDefaultLINEMOD();
    vector<string> classes;
    classes.push_back("06_template");
    detector.readClasses(classes, prefix + "/up/%s.yaml");

    vector<String> classes_ori;
    classes_ori.push_back("06_template");
    ori_detector->readClasses(classes_ori, prefix + "/up/%s.yaml");

    auto start_time = std::chrono::high_resolution_clock::now();
//    vector<cv::linemod::Match> matches;
//    ori_detector->match(sources, 70, matches, classes_ori);
    vector<linemodLevelup::Match> matches =
    detector.match(sources, 80, classes);
    auto match = matches[0];

    cout << "x: " << match.x << "\ny: " << match.y << "\ntemplate id: " << match.template_id
         << "\nsimilarity: "<< match.similarity <<endl;

    auto templ = detector.getTemplates(matches[0].class_id, matches[0].template_id);
    auto elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
    cout << "match time: " << elapsed_time.count()/1000000000.0 <<"s" << endl;
    cout << "break point line" << endl;
    return 0;
}
