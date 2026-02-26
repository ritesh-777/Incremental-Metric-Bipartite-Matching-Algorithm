#ifndef COMMON_STRUCTURES_H
#define COMMON_STRUCTURES_H

#include <vector>
#include <cmath>
#include <limits>
#include <cassert>
#include <iostream>
//#include <cuda_runtime.h>

// Constants and typedefs
typedef double Scalar;
const Scalar INF_np = std::numeric_limits<Scalar>::max();
const Scalar EPS_np = 1e-9;

// Point class for both 1D and 2D metrics
class Point_np {
public:
    std::vector<Scalar> coords;
    double x;
    double y;
    // Constructor with dimensions
    // Point(int dimensions) : coords(dimensions, 0) {}
    
    // Constructor with specific coordinates
    Point_np() : coords(1, 0) {}
    Point_np(const std::vector<Scalar>& coordinates) : coords(coordinates) {
        if(coords.size() ==2){
            x = coords[0];
            y = coords[1];
        }
    }
    
    // Distance function - Euclidean distance
    Scalar distance(const Point_np& other) const {
        if (coords.size() != other.coords.size()) {
            std::cerr
              << "ERROR: Point::distance dimension mismatch: "
              << "this has dim=" << coords.size()
              << ", other has dim=" << other.coords.size()
              << "\n";
            std::abort();   // stops with that message
        }
        Scalar sum = 0;
        for (size_t i = 0; i < coords.size(); ++i) {
            Scalar diff = coords[i] - other.coords[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // Distance function - Euclidean distance - GPU use
    /*__host__ __device__ Scalar distance(const Point_np& other) const {
        if (coords.size() != other.coords.size()) {
            std::cerr
              << "ERROR: Point::distance dimension mismatch: "
              << "this has dim=" << coords.size()
              << ", other has dim=" << other.coords.size()
              << "\n";
            std::abort();   // stops with that message
        }
        Scalar sum = 0;
        for (size_t i = 0; i < coords.size(); ++i) {
            Scalar diff = coords[i] - other.coords[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }*/

    bool operator==(const Point_np &o) const { return coords[0] == o.coords[0] && coords[1] == o.coords[1]; }
    bool operator<(const Point_np &o) const { return coords[0] < o.coords[0] || (coords[0] == o.coords[0] && coords[1] < o.coords[1]); }

};

// Edge class for admissible edges
struct Edge_np {
    int server_idx;
    int request_idx;
    Edge_np(int s, int r) : server_idx(s), request_idx(r) {}
};

#endif