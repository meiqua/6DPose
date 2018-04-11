#pragma once 

#include <vector>

struct Cluster {
    std::vector<double> mode;
    std::vector<std::vector<double> > original_points;
    std::vector<std::vector<double> > shifted_points;
};

class MeanShift {
public:
    typedef std::vector<double> Point;

    MeanShift() { set_kernel(nullptr); }
    MeanShift(double (*_kernel_func)(double,double)) { set_kernel(kernel_func); }
    std::vector<Point> meanshift(const std::vector<Point> & points,
                                                double kernel_bandwidth,
                                                double EPSILON = 0.00001);
    std::vector<Cluster> cluster(const std::vector<Point> &, double);

private:
    double (*kernel_func)(double,double);
    void set_kernel(double (*_kernel_func)(double,double));
    void shift_point(const Point&, const std::vector<Point> &, double, Point&);
    std::vector<Cluster> cluster(const std::vector<Point> &, const std::vector<Point> &);
};
