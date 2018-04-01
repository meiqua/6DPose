#include "linemodLevelup.h"
#include <memory>
#include <iostream>
#include "linemod_icp.h"
#include <assert.h>
using namespace std;
using namespace cv;

// for test
string type2str(int type) {
  string r;

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

static std::string prefix = "/home/meiqua/6DPose/linemodLevelup/test/case1/";

int main(){
    // test case1
    /*
     * template id: 1435
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
    vector<string> classes;
    classes.push_back("06_template");
    detector.readClasses(classes, prefix + "%s.yaml");

    auto matches = detector.match(sources, 50, classes);

    auto templ = detector.getTemplates(matches[0].class_id, matches[0].template_id);

    cout << "break point line" << endl;
    return 0;
}
