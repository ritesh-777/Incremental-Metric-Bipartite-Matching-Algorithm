#include "PushRelabel.h"
//#include "OnlineOptimal.h"
//#include "Greedy.h"
//#include "QT_Algo.h"
#include "DataReader.cpp"
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <omp.h>

// Function to generate random points within a bounding box
std::vector<Point_np> generateRandomPoints(int n, Scalar min_bound, Scalar max_bound, int dimensions, std::mt19937& gen) {
    std::uniform_real_distribution<Scalar> dist(min_bound, max_bound);
    std::vector<Point_np> points;
    for (int i = 0; i < n; ++i) {
        std::vector<Scalar> coords(dimensions);
        for (int j = 0; j < dimensions; ++j) {
            coords[j] = dist(gen);
        }
        points.push_back(Point_np(coords));
    }
    return points;
}

struct Result{
    std::vector<Point_np> server_locations;
    std::vector<Point_np> request_locations;
    // competitive ratios
    std::vector<Scalar> max_cr_opt;
    std::vector<Scalar> max_cr_pr;
    std::vector<Scalar> max_cr_greedy;
    std::vector<Scalar> max_cr_qt;
    //times
    std::vector<double> avg_pr_time;
    std::vector<double> avg_opt_time;
    std::vector<double> avg_greedy_time;
    std::vector<double> avg_qt_time;
    //pr specific times
    std::vector<double> pr_time_distance;
    std::vector<double> pr_time_find_admissible;
    std::vector<double> pr_time_dual_update;
    std::vector<double> pr_time_match_request;
};

// Function to run a single instance
Result* runInstance(int n, int num_requests, int inst, Scalar min_bound, Scalar max_bound, 
                                                        int dimensions, Scalar delta, std::mt19937& gen) {

    // Generate server points and request points for Taxi dataset
    DataSet d("PlotData/"+std::to_string(n)+"/"+std::to_string(inst+1)+"_Point_0.001000_Real_Eu_"+std::to_string(dimensions)+"dim"+".csv", num_requests);
    std::vector<Point_np> server_points = d.server_static();
    std::vector<Point_np> request_points = d.request_static();


    // Generate server points
    //std::vector<Point_np> server_points = generateRandomPoints(n, min_bound, max_bound, dimensions, gen);
    
    // Initialize all algorithms
    //Greedy greedyAlgo(server_points);
    /*Quadtree qt({min_bound, min_bound, max_bound, max_bound});
    for (const auto& point : server_points) {
        qt.insertServer({point.coords[0], point.coords[1]});
    }*/
    // Create a quadtree
    auto start_create_qt = std::chrono::high_resolution_clock::now();
    /*AABB<double> bounds = {min_bound, min_bound, max_bound, max_bound};
    Quadtree qt(bounds);
    // Insert points into the quadtree
    for (const auto& point : server_points) {
        qt.insertServer(point);
    }*/
    auto end_create_qt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> create_duration_qt = end_create_qt - start_create_qt;
    double qt_create_total_time = create_duration_qt.count();
    //OnlineMetricMatching optimalAlgo(server_points);
    OnlineDynamicMatching pushRelabel(dimensions, delta, min_bound, max_bound);
    for (const auto& point : server_points) {
        pushRelabel.add_server(point);
    }
    
    
    
    // Timing variables
    double pr_total_time = 0.0;
    double opt_total_time = 0.0;
    double greedy_total_time = 0.0;
    double qt_total_time = 0.0;

    // maximum competitive ratios
    Scalar max_cr_opt = 0.0;
    Scalar max_cr_pr = 0.0;
    Scalar max_cr_greedy = 0.0;
    Scalar max_cr_qt = 0.0;

    // PR specific times
    double pr_time_distance = 0.0;
    double pr_time_find_admissible = 0.0;
    double pr_time_dual_update = 0.0;
    double pr_time_match_request = 0.0;

    Result* result = new Result();

    result->server_locations = server_points;

    
    // Process requests
    for (int i = 0; i < num_requests; ++i) {
        // Read a taxi data
        Point_np request_point = request_points[i];

        // Generate a random request
        //Point_np request_point = generateRandomPoints(1, min_bound, max_bound, dimensions, gen)[0];

        result->request_locations.push_back(request_point);
        
        // Process with Push-Relabel algorithm and time it
        auto start_pr = std::chrono::high_resolution_clock::now();
        pushRelabel.match_request(request_point);
        auto end_pr = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_pr = end_pr - start_pr;
        //pr_total_time += duration_pr.count();
        pr_total_time = duration_pr.count();
        pr_time_distance = pushRelabel.get_time_distance();
        pr_time_find_admissible = pushRelabel.get_time_find_admissible();
        pr_time_dual_update = pushRelabel.get_time_dual_update();
        pr_time_match_request = pushRelabel.get_time_match_request();
        std::cout << std::endl;
        //pushRelabel.time_analysis();
        std::cout << "Request "<< i+1 << " is processed with cost " << pushRelabel.getTotalcost() << "." << std::endl;
        
        // Process with optimal algorithm and time it
        auto start_opt = std::chrono::high_resolution_clock::now();
        //optimalAlgo.processRequest(request_point);
        auto end_opt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_opt = end_opt - start_opt;
        //opt_total_time += duration_opt.count();
        opt_total_time = duration_opt.count();

        // Process with Greedy algorithm and time it
        auto start_greedy = std::chrono::high_resolution_clock::now();
        //greedyAlgo.processRequest(request_point);
        auto end_greedy = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_greedy = end_greedy - start_greedy;
        //greedy_total_time += duration_greedy.count();
        greedy_total_time = duration_greedy.count();

        // Process with Quadtree algorithm and time it
        auto start_qt = std::chrono::high_resolution_clock::now();
        //qt.matchRequest(request_point);
        auto end_qt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_qt = end_qt - start_qt;
        //qt_total_time += duration_qt.count();
        qt_total_time = duration_qt.count();

        // Calculate competitive ratios
        Scalar pr_cost = pushRelabel.getTotalcost();
        //Scalar greedy_cost = greedyAlgo.getTotalCost();
        //Scalar opt_cost = optimalAlgo.getTotalCost();
        //Scalar qt_cost = qt.getTotalcost();
        //max_cr_opt = opt_cost;
        max_cr_pr = pr_cost; // Avoid division by zero
        //max_cr_greedy = greedy_cost; // Avoid division by zero
        //max_cr_qt = qt_cost; // Avoid division by zero
        //max_cr_pr = std::max(max_cr_pr, pr_cost); // Avoid division by zero
        //max_cr_greedy = std::max(max_cr_greedy, greedy_cost); // Avoid division by zero
        //max_cr_qt = std::max(max_cr_qt, qt_cost); // Avoid division by zero
        /*max_cr_pr = std::max(max_cr_pr, pr_cost / std::max(opt_cost, EPS_np)); // Avoid division by zero
        max_cr_greedy = std::max(max_cr_greedy, greedy_cost / std::max(opt_cost, EPS_np)); // Avoid division by zero
        max_cr_qt = std::max(max_cr_qt, qt_cost / std::max(opt_cost, EPS_np)); // Avoid division by zero*/

        result->max_cr_opt.push_back(max_cr_opt);
        result->max_cr_pr.push_back(max_cr_pr);
        result->max_cr_greedy.push_back(max_cr_greedy);
        result->max_cr_qt.push_back(max_cr_qt);
        result->avg_pr_time.push_back(pr_total_time);
        result->avg_opt_time.push_back(opt_total_time);
        result->avg_greedy_time.push_back(greedy_total_time);
        if (i==0){
            result->avg_qt_time.push_back(qt_total_time+qt_create_total_time);
        }
        else{
            result->avg_qt_time.push_back(qt_total_time);
        }

        result->pr_time_distance.push_back(pr_time_distance);
        result->pr_time_find_admissible.push_back(pr_time_find_admissible);
        result->pr_time_dual_update.push_back(pr_time_dual_update);
        result->pr_time_match_request.push_back(pr_time_match_request);
        //std::cout<< "Request " << i << " is processed" << std::endl;
    }

    
    
    return result;
}

std::pair<Scalar, std::pair<double, double>> runInstance_no_opt(int n, int num_requests, Scalar min_bound, Scalar max_bound, 
                                                        int dimensions, Scalar delta, std::mt19937& gen) {
    
    // Generate server points and request points for Taxi dataset
    DataSet d("locations.csv", num_requests);
    std::vector<Point_np> server_points = d.server();
    std::vector<Point_np> request_points = d.request();
    
    // Generate server points
    //std::vector<Point_np> server_points = generateRandomPoints(n, min_bound, max_bound, dimensions, gen);
    
    // Initialize all algorithms
    //Greedy greedyAlgo(server_points);
    OnlineDynamicMatching pushRelabel(dimensions, delta, min_bound, max_bound);
    for (const auto& point : server_points) {
        pushRelabel.add_server(point);
    }
    
    
    // Timing variables
    double pr_total_time = 0.0;
    double greedy_total_time = 0.0;

    // maximum competitive ratios
    Scalar max_cr= 0.0;

    
    // Process requests
    for (int i = 0; i < num_requests; ++i) {
        // Read a taxi data
        Point_np request_point = request_points[i];

        // Generate a random request
        //Point_np request_point = generateRandomPoints(1, min_bound, max_bound, dimensions, gen)[0];
        
        // Process with Push-Relabel algorithm and time it
        auto start_pr = std::chrono::high_resolution_clock::now();
        pushRelabel.match_request(request_point);
        auto end_pr = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_pr = end_pr - start_pr;
        pr_total_time += duration_pr.count();

        // Process with Greedy algorithm and time it
        auto start_greedy = std::chrono::high_resolution_clock::now();
        //greedyAlgo.processRequest(request_point);
        auto end_greedy = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_greedy = end_greedy - start_greedy;
        greedy_total_time += duration_greedy.count();

        std::cout<< "Request " << i << " is processed" << std::endl;

        // Calculate competitive ratios
        Scalar pr_cost = pushRelabel.getTotalcost();
        //Scalar greedy_cost = greedyAlgo.getTotalCost();
        //max_cr = std::max(max_cr, pr_cost / std::max(greedy_cost, EPS_np)); // Avoid division by zero
    }

    return std::make_pair(max_cr, std::make_pair(pr_total_time / num_requests, greedy_total_time / num_requests));
}

int main() {
    const Scalar delta = 0.001;//0.125;
    const int dimensions = 2;
    const Scalar min_bound = 0;//0.0;
    const Scalar max_bound = 100.0;//100.0;
    const int num_instances = 10;
    //const std::vector<int> n_values = {1000};
    //const std::vector<int> n_values = {4200, 4400, 4600, 4800, 5000, 5200, 5400, 5600, 5800, 6000}; // Number of servers
    const std::vector<int> n_values = {10000};

    std::random_device rd;              // single rd visible to all threads
    //std::ofstream time_file("PlotData/Time_"+std::to_string(delta)+"_"+std::to_string(min_bound)+"_"+std::to_string(max_bound)+"Synthetic_"+std::to_string(dimensions)+"dim"+".csv");
    //cr_file   << "n,max_cr_pr,max_cr_greedy,max_cr_qt\n";
    //time_file << "n,pr_avg_time,greedy_avg_time,qt_time,opt_avg_time\n";

    /*for (int idx = 0; idx < (int)n_values.size(); ++idx) {
        for (int inst = 0; inst < num_instances; ++inst) {
            std::ofstream cr_file("PlotData/"+std::to_string(n_values[idx])+"/"+std::to_string(inst+1)+"CR_"+std::to_string(delta)+"_"+std::to_string(min_bound)+"_"+std::to_string(max_bound)+"Synthetic_"+std::to_string(dimensions)+"dim"+".csv");
        }
    }*/

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < (int)n_values.size(); ++idx) {
        int n = n_values[idx];

        // Scalar max_cr_pr = 0.0;
        // Scalar max_cr_greedy = 0.0;
        // Scalar max_cr_qt = 0.0;
        // double avg_pr_time = 0.0;
        // double avg_opt_time = 0.0;
        // double avg_greedy_time = 0.0;
        // double avg_qt_time = 0.0;

        std::vector<std::vector<Scalar>> max_cr_pr;
        std::vector<std::vector<Scalar>> max_cr_greedy;
        std::vector<std::vector<Scalar>> max_cr_qt;
        std::vector<std::vector<double>> avg_pr_time;
        std::vector<std::vector<double>> avg_opt_time;
        std::vector<std::vector<double>> avg_greedy_time;
        std::vector<std::vector<double>> avg_qt_time;

        // parallelize the instances for this n
        #pragma omp parallel for schedule(dynamic) 
        // reduction(max: max_cr_pr) 
        // reduction(max: max_cr_greedy) 
        // reduction(max: max_cr_qt) 
        // reduction(+: avg_pr_time, avg_opt_time, avg_greedy_time, avg_qt_time)
        for (int inst = 0; inst < num_instances; ++inst) {
            // Seed each thread’s RNG with rd plus the thread+instance index
            std::mt19937 local_gen(rd() + omp_get_thread_num()*1000 + inst);

            // Scalar max_cr_pr = 0.0;
            // Scalar max_cr_greedy = 0.0;
            // Scalar max_cr_qt = 0.0;
            // double avg_pr_time = 0.0;
            // double avg_opt_time = 0.0;
            // double avg_greedy_time = 0.0;
            // double avg_qt_time = 0.0;

            auto result = runInstance(n, n, inst, min_bound, max_bound, dimensions, delta, local_gen);
            /*max_cr_pr = std::max(max_cr_pr, result->max_cr_pr);
            max_cr_greedy = std::max(max_cr_greedy, result->max_cr_greedy);
            max_cr_qt = std::max(max_cr_qt, result->max_cr_qt);
            avg_pr_time += result->avg_pr_time;
            avg_greedy_time += result->avg_greedy_time;
            avg_opt_time += result->avg_opt_time;
            avg_qt_time += result->avg_qt_time;*/
            // max_cr_pr.push_back(result->max_cr_pr);
            // max_cr_greedy.push_back(result->max_cr_greedy);
            // max_cr_qt.push_back(result->max_cr_qt);
            // avg_pr_time.push_back(result->avg_pr_time);
            // avg_greedy_time.push_back(result->avg_greedy_time);
            // avg_opt_time.push_back(result->avg_opt_time);
            // avg_qt_time.push_back(result->avg_qt_time);

            //std::ofstream point_file("PlotData/"+std::to_string(n_values[idx])+"/"+std::to_string(inst+1)+"_Point_"+std::to_string(delta)+"_Synt_Eu_"+std::to_string(dimensions)+"dim"+".csv");
            std::ofstream pr_file("PlotData/"+std::to_string(n_values[idx])+"/"+std::to_string(inst+1)+"_PR_"+std::to_string(delta)+"_Real_Eu_"+std::to_string(dimensions)+"dim"+".csv");
            //std::ofstream greedy_file("PlotData/"+std::to_string(n_values[idx])+"/"+std::to_string(inst+1)+"_Greedy_"+std::to_string(delta)+"_Synt_Eu_"+std::to_string(dimensions)+"dim"+".csv");
            //std::ofstream opt_file("PlotData/"+std::to_string(n_values[idx])+"/"+std::to_string(inst+1)+"_Opt_"+std::to_string(delta)+"_Synt_Eu_"+std::to_string(dimensions)+"dim"+".csv");
            //std::ofstream qt_file("PlotData/"+std::to_string(n_values[idx])+"/"+std::to_string(inst+1)+"_QT_"+std::to_string(delta)+"_Synt_Eu_"+std::to_string(dimensions)+"dim"+".csv");
            //std::ofstream cr_file("PlotData/"+std::to_string(n_values[idx])+"/"+std::to_string(inst+1)+"_CR_"+std::to_string(delta)+"_Real_Eu_"+std::to_string(dimensions)+"dim"+".csv");
            //std::ofstream time_file("PlotData/"+std::to_string(n_values[idx])+"/"+std::to_string(inst+1)+"_Time_"+std::to_string(delta)+"_Real_Eu_"+std::to_string(dimensions)+"dim"+".csv");

            //point_file << "server_x,server_y,request_x,request_y\n";
            pr_file << "n,cost,run_time,time_computing_distances,time_find_admissible,time_dual_updates,time_update_matching\n";
            //greedy_file << "n,cost,run_time\n";
            //opt_file << "n,cost,run_time\n";
            //qt_file << "n,cost,run_time\n";
            //cr_file << "n,max_cr_pr,max_cr_greedy,max_cr_qt\n";
            //time_file << "n,pr_avg_time,greedy_avg_time,qt_time,opt_avg_time\n";

            for (int i = 0; i < result->server_locations.size(); ++i) {
                //std::cout   << i+1 << "," << result->max_cr_pr[i] << "," << result->max_cr_greedy[i]  << "," << result->max_cr_qt[i]<<"\n";
                //std::cout << i+1 << "," << result->avg_pr_time[i] <<","<< result->avg_greedy_time[i]  << "," << result->avg_qt_time[i]<<","<< result->avg_opt_time[i] << "\n";
                
                //point_file << result->server_locations[i].coords[0] << "," << result->server_locations[i].coords[1]  << "," << result->request_locations[i].coords[0] << "," << result->request_locations[i].coords[1]<<"\n";
                pr_file   << i+1 << "," << result->max_cr_pr[i] << "," << result->avg_pr_time[i] << "," << result->pr_time_distance[i] << "," << result->pr_time_find_admissible[i] << "," << result->pr_time_dual_update[i] << "," << result->pr_time_match_request[i] << "\n";
                //greedy_file   << i+1 << "," << result->max_cr_greedy[i] << "," << result->avg_greedy_time[i]  << "\n";
                //opt_file   << i+1 << "," << result->max_cr_opt[i] << "," << result->avg_opt_time[i]  << "\n";
                //qt_file   << i+1 << "," << result->max_cr_qt[i] << "," << result->avg_qt_time[i]  << "\n";
                //time_file << i+1 << "," << result->avg_pr_time[i] <<","<< result->avg_greedy_time[i]  << "," << result->avg_qt_time[i]<<","<< result->avg_opt_time[i] << "\n";

                //cr_file   << i+1 << "," << result->max_cr_pr[i] << "," << result->max_cr_greedy[i]  << "," << result->max_cr_qt[i]<<"\n";
                //time_file << i+1 << "," << result->avg_pr_time[i] <<","<< result->avg_greedy_time[i]  << "," << result->avg_qt_time[i]<<","<< result->avg_opt_time[i] << "\n";
            }

            //point_file.close();
            pr_file.close();
            //greedy_file.close();
            //opt_file.close();
            //qt_file.close();

            delete result; // Free the allocated memory
            
            std::cout << "Instance " << inst+1 << " of n = " << n_values[idx] << " in Taxi with delta = " << delta << " is done." << std::endl;
        }

        std::cout << "\nn_value = " << n_values[idx] << " is done.\n" << std::endl;

        // Calculate average times
        // double avg_pr = avg_pr_time / num_instances;
        // double avg_opt = avg_opt_time / num_instances;
        // double avg_greedy = avg_greedy_time / num_instances;
        // double avg_qt = avg_qt_time / num_instances;


        // serialize file I/O
        // #pragma omp critical
        // {

        //     cr_file   << n << "," << max_cr_pr << "," << max_cr_greedy  << "," << max_cr_qt<<"\n";
        //     time_file << n << "," << avg_pr <<","<<avg_greedy  << "," << avg_qt<<","<<avg_opt << "\n";
        //     std::cout  << "n=" << n << "  maxCR_pushRelabel=" << max_cr_pr << "  macCR_greedy="<<max_cr_greedy
        //                << "  maxCR_qt=" << max_cr_qt
        //                << "  pr_time=" << avg_pr << "ms"
        //                << "  greedy_time=" << avg_greedy << "ms"
        //                << "  qt_time=" << avg_qt << "ms"
        //                << "  opt_time=" << avg_opt << "ms\n";
        // }
    }

    // #pragma omp parallel for schedule(dynamic)
    // for (int idx = 0; idx < (int)n_values.size(); ++idx) {
    //     int n = n_values[idx];

    //     Scalar max_cr = 0.0;
    //     double avg_pr_time = 0.0;
    //     double avg_greedy_time = 0.0;

    //     // parallelize the instances for this n
    //     #pragma omp parallel for schedule(dynamic) \
    //     reduction(max: max_cr) \
    //     reduction(+: avg_pr_time, avg_greedy_time)
    //     for (int inst = 0; inst < num_instances; ++inst) {
    //         // Seed each thread’s RNG with rd plus the thread+instance index
    //         std::mt19937 local_gen(rd() + omp_get_thread_num()*1000 + inst);

    //         auto result = runInstance_no_opt(n, n, min_bound, max_bound, dimensions, delta,local_gen);
    //         max_cr = std::max(max_cr, result.first);
    //         avg_pr_time += result.second.first;
    //         avg_greedy_time += result.second.second;  
    //     }

    //     // Calculate average times
    //     double avg_pr = avg_pr_time / num_instances;
    //     double avg_greedy = avg_greedy_time / num_instances;

    //     // serialize file I/O
    //     #pragma omp critical
    //     {
    //         cr_file   << n << "," << max_cr << "\n";
    //         time_file << n << "," << avg_pr <<","<<avg_greedy  << "\n";
    //         std::cout  << "n=" << n << "  maxCR_pr/greedy=" << max_cr
    //                    << "  pr_time=" << avg_pr << "ms"
    //                    << "  greedy_time=" << avg_greedy << "ms"<<std::endl;
    //     }
    // }

    // cr_file.close();
    // time_file.close();
    return 0;
}
