#include "QT_Algo.h"
#include "DataReader.cpp"
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <omp.h>

struct Result{
    std::vector<Point_np> server_locations;
    std::vector<Point_np> request_locations;
    // competitive ratios
    std::vector<Scalar> max_cr_qt;
    //times
    std::vector<double> avg_qt_time;
};

// Function to run a single instance
Result* runInstance(int n, int num_requests, int inst, Scalar min_bound, Scalar max_bound, 
                                                        int dimensions, Scalar delta, std::mt19937& gen) {

    // Generate server points and request points for Taxi dataset
    DataSet d("PlotData/"+std::to_string(n)+"/"+std::to_string(inst+1)+"_Point_"+std::to_string(delta)+"_Synt_Eu_"+std::to_string(dimensions)+"dim"+".csv", num_requests);
    std::vector<Point_np> server_points = d.server_static();
    std::vector<Point_np> request_points = d.request_static();


    auto start_create_qt = std::chrono::high_resolution_clock::now();
    AABB<double> bounds = {min_bound, min_bound, max_bound, max_bound};
    Quadtree qt(bounds);
    // Insert points into the quadtree
    for (const auto& point : server_points) {
        qt.insertServer(point);
    }
    auto end_create_qt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> create_duration_qt = end_create_qt - start_create_qt;
    double qt_create_total_time = create_duration_qt.count();
    
    
    
    // Timing variables
    double qt_total_time = 0.0;

    // maximum competitive ratios
    Scalar max_cr_qt = 0.0;

    Result* result = new Result();

    result->server_locations = server_points;

    
    // Process requests
    for (int i = 0; i < num_requests; ++i) {
        // Read a synthetic data
        Point_np request_point = request_points[i];


        result->request_locations.push_back(request_point);
        
        // Process with Quadtree algorithm and time it
        auto start_qt = std::chrono::high_resolution_clock::now();
        qt.matchRequest(request_point);
        auto end_qt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_qt = end_qt - start_qt;
        //qt_total_time += duration_qt.count();
        qt_total_time = duration_qt.count();

        // Calculate competitive ratios
        Scalar qt_cost = qt.getTotalcost();
        max_cr_qt = qt_cost;

        result->max_cr_qt.push_back(max_cr_qt);
        if (i==0){
            result->avg_qt_time.push_back(qt_total_time+qt_create_total_time);
        }
        else{
            result->avg_qt_time.push_back(qt_total_time);
        }
        //std::cout<< "Request " << i << " is processed" << std::endl;
    }

    
    
    return result;
}


int main() {
    const Scalar delta = 0.001;
    const int dimensions = 2;
    const Scalar min_bound = 0;//0.0;
    const Scalar max_bound = 100.0;//100.0;
    const int num_instances = 10;

    const std::vector<int> n_values = {10000};

    std::random_device rd;              // single rd visible to all threads
    

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < (int)n_values.size(); ++idx) {
        int n = n_values[idx];

        
        std::vector<std::vector<Scalar>> max_cr_qt;
        std::vector<std::vector<double>> avg_qt_time;

        // parallelize the instances for this n
        #pragma omp parallel for schedule(dynamic) 
        
        for (int inst = 0; inst < num_instances; ++inst) {
            // Seed each thread’s RNG with rd plus the thread+instance index
            std::mt19937 local_gen(rd() + omp_get_thread_num()*1000 + inst);

            auto result = runInstance(n, n, inst, min_bound, max_bound, dimensions, delta, local_gen);
           
            std::ofstream qt_file("PlotData/"+std::to_string(n_values[idx])+"/"+std::to_string(inst+1)+"_QT_"+std::to_string(delta)+"_Synt_Eu_"+std::to_string(dimensions)+"dim"+".csv");
            
            qt_file << "n,cost,run_time\n";

            for (int i = 0; i < result->server_locations.size(); ++i) {
                
                qt_file   << i+1 << "," << result->max_cr_qt[i] << "," << result->avg_qt_time[i]  << "\n";
            }

            

            delete result; // Free the allocated memory
            
            std::cout << "Instance " << inst+1 << " of n = " << n_values[idx] << " in synt with delta = " << delta << " is done." << std::endl;
        }

        std::cout << "\nn_value = " << n_values[idx] << " is done.\n" << std::endl;

    }

   
    return 0;
}
