#ifndef COMMON_STRUCTURES_H
#define COMMON_STRUCTURES_H

#include <vector>
#include <cmath>
#include <limits>
#include <cassert>
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iomanip>

#include "dist_cache.hpp"

// Constants and typedefs
typedef long double Scalar;
const Scalar INF_np = std::numeric_limits<Scalar>::max();
const Scalar EPS_np = 1e-9;

// Point class for both 1D and 2D metrics
class Point_np {
public:
    std::vector<Scalar> coords;
    double x;
    double y;

    // Default to 2D (safer if most code expects 2 components)
    Point_np() : coords(2, 0.0), x(0.0), y(0.0) {}

    Point_np(const std::vector<Scalar>& coordinates) : coords(coordinates) {
        if (coords.size() >= 1) {
            x = static_cast<double>(coords[0]);
        } else {
            x = 0.0;
        }
        if (coords.size() >= 2) {
            y = static_cast<double>(coords[1]);
        } else {
            y = 0.0;
        }
    }

    // helper: format a Scalar to the exact decimal integer string if it is an integer
    std::string scalar_to_nodeid_string(Scalar s) const {
        // if s is very close to an integer, format as integer to match nodes.txt
        long long as_int = static_cast<long long>(s);
        if (std::fabs(s - static_cast<Scalar>(as_int)) < 1e-6) {
            return std::to_string(as_int);
        }
        // otherwise, print full precision (no scientific) — adjust precision if needed
        std::ostringstream oss;
        oss.setf(std::ios::fmtflags(0), std::ios::floatfield); // default formatting
        oss << std::fixed << std::setprecision(6) << s;         // pick precision suitable for your IDs
        return oss.str();
    }

    // Compute shortest-path distance using DistCache.
    // This uses coords[0] as the numeric node id for this point and other.coords[0] for the target.
    // If your node-id mapping is different, adapt this method.
    Scalar distance_shortest_path(const Point_np& other, DistCache& cache, bool verbose = false) const {
        if (coords.empty() || other.coords.empty()) {
            std::cerr << "ERROR: Point::distance_shortest_path given empty coords\n";
            std::abort();
        }

        // Convert the first coordinate of each point to node-id string
        Scalar src_s = coords[0];
        Scalar tgt_s = other.coords[0];

        std::string src_id = scalar_to_nodeid_string(src_s);
        std::string tgt_id = scalar_to_nodeid_string(tgt_s);

        if (verbose) {
            std::cout << "Looking up distance from '" << src_id << "' to '" << tgt_id << "' in DistCache\n";
        }

        float d = cache.get_distance_from_cache(src_id, tgt_id, verbose);

        // convert float -> Scalar (double)
        if (std::isinf(d)) {
            return std::numeric_limits<Scalar>::infinity();
        }
        //std::cout << "Distance from " << src_id << " to " << tgt_id << " is " << d << std::endl;
        return static_cast<Scalar>(d);
    }

    // Comparisons made robust to 1D/2D points
    bool operator==(const Point_np &o) const {
        if (coords.size() != o.coords.size()) {
            return false;
        }
        for (size_t i = 0; i < coords.size(); ++i) {
            if (std::fabs(coords[i] - o.coords[i]) > EPS_np) return false;
        }
        return true;
    }

    bool operator<(const Point_np &o) const {
        // lexicographic compare for any dimension
        size_t m = std::min(coords.size(), o.coords.size());
        for (size_t i = 0; i < m; ++i) {
            if (coords[i] < o.coords[i]) return true;
            if (coords[i] > o.coords[i]) return false;
        }
        // shorter vector is 'less' if all equal so far
        return coords.size() < o.coords.size();
    }

};

// Edge class for admissible edges
struct Edge_np {
    int server_idx;
    int request_idx;
    Edge_np(int s, int r) : server_idx(s), request_idx(r) {}
};

#endif // COMMON_STRUCTURES_H
