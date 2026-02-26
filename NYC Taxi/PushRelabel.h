#ifndef PushRelabel_H
#define PushRelabel_H

#include <iostream>
#include <tuple>
#include <chrono>
#include "common_structures.h"
#include "OnlineOptimal.h"

//Scalar curr_omega = 0.0;

// Request class
class Request_np {
    public:
        Point_np location;
        std::vector<int> y; // Dual weights at each level
        int level; // Current level of the request
        
        Request_np(const Point_np& loc, int maxLevel) 
            : location(loc), y(maxLevel + 2, 0), level(0) {}

        //void reset_level() { level = 0; }
        //void reset_y() { std::fill(y.begin(), y.end(), 0); }
    };
    
// Server class
class Server_np {
public:
    Point_np location;
    std::vector<int> y; // Dual weights at each level
    int matched_request_idx; // Index of the request this server is matched to (-1 if unmatched)
    
    Server_np(const Point_np& loc, int maxLevel) 
        : location(loc), y(maxLevel + 2, 0), matched_request_idx(-1) {}

    void reset_y() { std::fill(y.begin(), y.end(), 0); }
    void reset_matched_request() { matched_request_idx = -1; }
};
    

// Algorithm state class for maintaining multiple solutions
class AlgorithmState {
public:
    std::vector<Server_np> servers;
    std::vector<Request_np> requests;
    Scalar omega; // The guess for optimal matching cost
    
    AlgorithmState(const std::vector<Server_np>& s, Scalar w) : omega(w) {
        servers = s;
    }
    
    // Add a request to this state
    void add_request(const Request_np& r) {
        requests.push_back(r);
    }

    Scalar get_number_of_requests_level(int level) const {
        Scalar count = 0;
        for (const Request_np& req : requests) {
            if (req.level >= level) {
                count++;
            }
        }
        return count;
    }

    Scalar get_max_level_of_requests() const {
        Scalar max_level = 0;
        for (const Request_np& req : requests) {
            if (req.level > max_level) {
                max_level = req.level;
            }
        }
        return max_level;
    }

    void set_omega(Scalar new_omega) {
        omega = new_omega;
    }
    
    // Get the total cost of the current matching
    Scalar getTotalcost() const {
        Scalar cost = 0;
        for(const Server_np& server : servers) {
            if (server.matched_request_idx != -1) {
                cost += server.location.distance(requests[server.matched_request_idx].location);
            }
        }
        return cost;
    }
};

// Main algorithm class
class OnlineDynamicMatching {
private:
    // Parameters
    Scalar min_bound;
    Scalar max_bound;
    std::vector<Server_np> initial_servers;
    int dimensions; // Dimensionality of the metric space
    int mu; // Number of levels
    Scalar delta; // Parameter from the paper
    Scalar epsilon; // Parameter from the paper
    std::vector<AlgorithmState> solutions; // Multiple solutions for different omega values
    //std::vector<AlgorithmState> solution_state; // Single solutions for different omega values
    Scalar curr_omega = 1.0; // Current best omega value
    int num_opt_servers = 0;
    bool inside_opt = false;
    OnlineMetricMatching optimalAlgo;

    double time_distance = 0.0;
    double time_find_admissible = 0.0;
    double time_dual_update = 0.0;
    double time_match_request = 0.0;


    std::vector<Point_np> optimal_server(AlgorithmState& state) {
    
        std::vector<Point_np> servers;
        for (size_t s = 0; s < state.servers.size(); ++s) {
            Server_np& server = state.servers[s];
            if (server.matched_request_idx == -1) {
                servers.push_back(server.location);
            }
        }
        return servers;

    }
    
    // Helper functions
    Scalar scaled_distance(int level, const Point_np& s, const Point_np& r, Scalar omega) {
        if (level == 0) {
            return std::ceil(2 * s.distance(r) * initial_servers.size() / (epsilon * omega));
        } else {
            return std::ceil(scaled_distance(level - 1, s, r, omega) / 
                           (2 * std::pow(1 + epsilon, 2) * std::pow(initial_servers.size(), phi(level - 1))));
        }
    }

    Scalar big_phi(int level) {
        return ceil(((std::pow(3, level) - 1) / 2) * (2 * delta));
    }
    
    Scalar phi(int i) {
        return std::pow(3, i) * (2 * delta);
    }
    
    Scalar y_max(int level) {
        return (25 / epsilon) * std::pow(initial_servers.size(), phi(level));
    }
    
    // Find an admissible edge for a free request at a given level
    Edge_np find_admissible_edge(AlgorithmState& state, int request_idx, int level) {
        Request_np& r = state.requests[request_idx];
        int matched_server = -1;
        Scalar min_slack = INF_np;
        
        for (size_t s = 0; s < state.servers.size(); ++s) {
            Server_np& server = state.servers[s];
            //auto start_time_distance = std::chrono::high_resolution_clock::now();
            Scalar d_i = scaled_distance(level, server.location, r.location, state.omega);
            //auto end_time_distance = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double, std::milli> duration_distance = end_time_distance - start_time_distance;
            //time_distance += duration_distance.count();
            
            if(server.matched_request_idx == -1 || level <= state.requests[server.matched_request_idx].level) {
                min_slack = std::min(min_slack, d_i - r.y[level] - server.y[level]);
            }

            if (server.y[level] + r.y[level] == d_i + 1.0) {
                if (server.matched_request_idx == -1) {
                    // std::cout << "Admissible edge found: Server_np " << s << ", Request " << request_idx << std::endl;
                    return Edge_np(s, request_idx);
                }
                else if (level <= state.requests[server.matched_request_idx].level && matched_server == -1) {
                    // std::cout << "Admissible edge found: Server_np " << s << ", Request " << request_idx << std::endl;
                    //return Edge_np(s, request_idx);
                    matched_server = s;
                }
            }
        }

        /*if (matched_server != -1){

        }*/
        
        //return Edge_np(-1, -1); // No admissible edge found
        return Edge_np(matched_server, (int)min_slack); // No admissible edge found
    }

    /*std::tuple<Scalar, size_t> find_closest_server_distance(AlgorithmState& state, int request_idx, int level) {
        Request_np& r = state.requests[request_idx];

        Scalar MIN = std::numeric_limits<Scalar>::max();
        size_t server_location;
        
        for (size_t s = 0; s < state.servers.size(); ++s) {
            Server_np& server = state.servers[s];
            Scalar d_i = scaled_distance(level, server.location, r.location, state.omega);
            if (d_i + 1.0 - server.y[level] - r.y[level] < MIN) {
                // std::cout << "Admissible edge found: Server_np " << s << ", Request " << request_idx << std::endl;
                MIN = d_i + 1.0 - server.y[level] - r.y[level];
                server_location = s;
            }
        }
        
        return std::make_tuple(MIN,server_location); // No admissible edge found
    }*/
    
    // Match a request in a specific state
    void match_request_in_state(AlgorithmState& state, const Point_np& location) {
        // Add the new request to this state
        Request_np new_request(location, mu);
        int new_request_idx = state.requests.size();
        state.add_request(new_request);
        
        int free_request_idx = new_request_idx;
        // set the dual-0 of new request y0(rf ) ← min ymax0 , min s∈S {di(rf , s) + 1 − y0(s)}
        //auto start_time_distance = std::chrono::high_resolution_clock::now();
        state.requests[free_request_idx].y[0] = y_max(0);
        for (size_t s = 0; s < state.servers.size(); ++s) {
            Server_np& server = state.servers[s];
            int d_i = scaled_distance(0, server.location, new_request.location, state.omega);
            state.requests[free_request_idx].y[0] = std::min(state.requests[free_request_idx].y[0], d_i + 1 - server.y[0]);
        }
        //auto end_time_distance = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double, std::milli> duration_distance = end_time_distance - start_time_distance;
        //time_distance += duration_distance.count();

        int level = 0;
        
        while (free_request_idx != -1) {
            //Scalar closest_server_distance;
            //size_t closest_server_location;
            //std::cout << "Level: " << level << ", Free Request_np Index: " << free_request_idx << " y value :" << state.requests[free_request_idx].y[level] << std::endl;
            //std::cin.get();
            // std::cout << "Level: " << level << ", Free Request_np Index: " << free_request_idx << std::endl;
            Request_np& r_free = state.requests[free_request_idx];
            
            // Check if the dual reached y_max
            if (r_free.y[level] >= y_max(level)) {
                // std::cout << "Dual reached y_max at level " << level << std::endl;  
                r_free.y[level] = y_max(level);
                level++;
                if (level == mu+2){
                    exit(0);
                }
                r_free.level = level;
                continue;
            }
            
            // Find admissible edge
            //auto start_time_find_admissible = std::chrono::high_resolution_clock::now();
            Edge_np e = find_admissible_edge(state, free_request_idx, level);
            //auto end_time_find_admissible = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double, std::milli> duration_find_admissible = end_time_find_admissible - start_time_find_admissible;
            //time_find_admissible += duration_find_admissible.count();
            
            if (e.server_idx != -1) { // Found an admissible edge
                Server_np& s = state.servers[e.server_idx];
                //std::cout<<"In every 2 step we are at admissinble"<<std::endl;
                
                if (s.matched_request_idx == -1) { // Admissible with free server
                    //auto start_time_match_request = std::chrono::high_resolution_clock::now();
                    s.matched_request_idx = free_request_idx;
                    //auto start_time_dual_update = std::chrono::high_resolution_clock::now();
                    s.y[level] -= 1; // Adjust dual weight
                    //auto end_time_dual_update = std::chrono::high_resolution_clock::now();
                    //std::chrono::duration<double, std::milli> duration_dual_update = end_time_dual_update - start_time_dual_update;
                    //time_dual_update += duration_dual_update.count();
                    free_request_idx = -1; // No more free request
                    //auto end_time_match_request = std::chrono::high_resolution_clock::now();
                    //std::chrono::duration<double, std::milli> duration_match_request = end_time_match_request - start_time_match_request;
                    //time_match_request += duration_match_request.count();

                    //std::cout << "Admissible with free server"<< std::endl;
                } else { // Admissible with matched server
                    //auto start_time_match_request = std::chrono::high_resolution_clock::now();
                    int old_request_idx = s.matched_request_idx;
                    
                    s.matched_request_idx = free_request_idx;
                    //auto start_time_dual_update = std::chrono::high_resolution_clock::now();
                    s.y[level] -= 1; // Adjust dual weight
                    //auto end_time_dual_update = std::chrono::high_resolution_clock::now();
                    //std::chrono::duration<double, std::milli> duration_dual_update = end_time_dual_update - start_time_dual_update;
                    //time_dual_update += duration_dual_update.count();
                    free_request_idx = old_request_idx; // Now the old request is free
                    level = state.requests[free_request_idx].level;
                    //auto end_time_match_request = std::chrono::high_resolution_clock::now();
                    //std::chrono::duration<double, std::milli> duration_match_request = end_time_match_request - start_time_match_request;
                    //time_match_request += duration_match_request.count();
                    //std::cout << "Admissible with non-free server"<< std::endl;
                }
            } else { // No admissible edge
                //auto start_time_dual_update = std::chrono::high_resolution_clock::now();
                r_free.y[level] += (e.request_idx + 1); // Increase dual weight
                //auto end_time_dual_update = std::chrono::high_resolution_clock::now();
                //std::chrono::duration<double, std::milli> duration_dual_update = end_time_dual_update - start_time_dual_update;
                //time_dual_update += duration_dual_update.count();
                //std::cout << "Increase in request dual with amount " << e.request_idx << "\n";
                

                //std::tie(closest_server_distance,closest_server_location) = find_closest_server_distance (state, free_request_idx, level);
                //r_free.y[level] = r_free.y[level] + closest_server_distance;
                //std::cout << "not-Admissible"<< std::endl;
            }
        }
    }
    
public:
    OnlineDynamicMatching(int dims, Scalar d = 0.25, Scalar min_val = 0, Scalar max_val = 1) 
        : dimensions(dims), delta(d), num_opt_servers(50) {
        mu = std::ceil(std::log(2.0/delta) / std::log(3.0)) + 1;
        epsilon = 1.0 / (std::log(1.0/delta) / std::log(3.0));

        // compute original upper bound on mu from delta
        int mu_orig = (int) std::ceil(std::log((2.0/(9*delta)) - 1.0) / std::log(3.0));

        // override epsilon for experiments (choose <= 0.5)
        epsilon = 0.2;   // <-- set the epsilon you want (e.g., 0.2 or 0.5)

        // choose mu so epsilon \u2248 1/mu but do not exceed mu_orig
        int mu_from_eps = std::max(1, (int)std::floor(1.0 / epsilon));
        mu = std::min(mu_orig, mu_from_eps);

        min_bound = min_val;
        max_bound = max_val;
        


    }
    
    // Add a server
    void add_server(const Point_np& location) {
        initial_servers.push_back(Server_np(location, mu));
    }

    
    // Initialize solutions with different omega values
    void initialize_solutions() {
        /*solutions.clear();
        //
        solutions.push_back(AlgorithmState(initial_servers, 0.0)); // Initial solution with omega = 0
        
        // Initialize solutions with different omega values
        int n_solutions = std::ceil(std::log2(initial_servers.size() * max_dist)) + 1;
        for (int i = 0; i < n_solutions; ++i) {
            Scalar omega = std::pow(2, i);
            solutions.push_back(AlgorithmState(initial_servers, omega));
        }*/

        solutions.clear();
        solutions.push_back(AlgorithmState(initial_servers, 1.0));
        solutions.push_back(AlgorithmState(initial_servers, 1.0));
    }


    bool verify_solution(const AlgorithmState& state) {
        // Check if the solution is valid
        Scalar max_level = state.get_max_level_of_requests();
        for(int level = 0; level <= max_level; ++level) {
            Scalar total_requests = state.get_number_of_requests_level(level);
            Scalar possible_requests = std::pow(state.servers.size(), 1 - big_phi(level));
            if (total_requests > possible_requests) {
                return false;
            }
        }
        return true;
    }

    void clean_solution(AlgorithmState& state) {
        // Clean the solution by removing requests that are not matched
        /*for (size_t i = 0; i < state.requests.size(); ++i) {
            state.requests[i].reset_level();
            state.requests[i].reset_y();
        }*/
        // Clean the solution by removing all requests
        state.requests.clear();
        for (size_t i = 0; i < state.servers.size(); ++i) {
            state.servers[i].reset_matched_request();
            state.servers[i].reset_y();
        }
    }
    
    // Match a new request
    void match_request(const Point_np& location) {
        // If this is the first request, we need to initialize the solutions
        if (solutions.empty()) {
            // Initialize solutions with different omega values
            //Scalar del = dimensions == 1 ? 1 : std::sqrt(2);
            initialize_solutions();
        }

        time_distance = 0.0;
        time_find_admissible = 0.0;
        time_dual_update = 0.0;
        time_match_request = 0.0;
        
        // Match the request in each solution
        /*for (AlgorithmState& state : solutions) {
            match_request_in_state(state, location);
        }*/

        if (solutions[0].servers.size()-solutions[0].requests.size() == num_opt_servers || inside_opt){

            if (solutions[0].servers.size()-solutions[0].requests.size() == num_opt_servers) {
                optimalAlgo.loadServers(optimal_server(solutions[0]));
                inside_opt = true;
            }
            
            optimalAlgo.processRequest(location);
        }
        else {
            // Update the solution state
            solutions[1] = solutions[0];

            do
            {
                //std::cout << "solution varification " << verify_solution(solution_state[1]) << std::endl;
                if (verify_solution(solutions[1]) == false) {
                    curr_omega = curr_omega * 2;
                    //std::cout << " bef curr_omega: " << curr_omega << std::endl;
                    clean_solution(solutions[1]);
                    //std::cout << " bef 2 curr_omega: " << curr_omega << std::endl;
                    solutions[1].set_omega(curr_omega);
                    //std::cout << " bef 3 curr_omega: " << curr_omega << std::endl;
                    for (int i = 0; i < solutions[0].requests.size(); ++i) {
                        match_request_in_state(solutions[1], solutions[0].requests[i].location);
                    }
                    //std::cout << " bef 4 curr_omega: " << curr_omega << std::endl;
                    match_request_in_state(solutions[1], location);
                    //std::cout << "curr_omega: " << curr_omega << std::endl;
                }
                else
                {
                    solutions[1].set_omega(curr_omega);
                    match_request_in_state(solutions[1], location);
                /* code */
                }
                //std::cout << "curr_omega: here" << curr_omega << verify_solution(solution_state[1]) << std::endl;
            }while(verify_solution(solutions[1]) == false);

            solutions[0] = solutions[1];

            /*if (solutions[0].servers.size()-solutions[0].requests.size() <= num_opt_servers && !inside_opt) {
                optimalAlgo.loadServers(optimal_server(solutions[0]));
                inside_opt = true;
            }*/
        }




        

        /*for (int i = curr_omega; i < solutions.size(); ++i) {
            match_request_in_state(solutions[i], location);
        }*/

        /*if (!solutions.empty()) {
            curr_omega = set_current_omega();
        }*/
    }

    void time_analysis() const {
        std::cout << "Time Analysis (in milliseconds):" << std::endl;
        std::cout << "Distance Calculations Time: " << time_distance << " ms" << std::endl;
        std::cout << "Finding Admissible Edges Time: " << time_find_admissible << " ms" << std::endl;
        std::cout << "Dual Updates Time: " << time_dual_update << " ms" << std::endl;
        std::cout << "Matching Requests Time: " << time_match_request << " ms" << std::endl;
    }

    double get_time_distance() const {
        return time_distance;
    }

    double get_time_find_admissible() const {
        return time_find_admissible;
    }

    double get_time_dual_update() const {
        return time_dual_update;
    }

    double get_time_match_request() const {
        return time_match_request;
    }


    /*Scalar set_current_omega() const {
        Scalar min_cost = INF_np;
        int best_idx = 0;
        
        for (size_t i = curr_omega; i < solutions.size(); ++i) {
            Scalar cost = solutions[i].get_cost();
            if (cost < min_cost) {
                min_cost = cost;
                best_idx = i;
            }
        }

        std::cout << "Working omega: " << solutions[best_idx].omega << "  Current omega: " << curr_omega << std::endl;
        if (solutions[best_idx].omega > curr_omega) {
            return solutions[best_idx].omega;
        }
        else {
            return curr_omega;
        }
    }*/


    
    // Get the best matching (with lowest cost)
    /*std::vector<Edge_np> get_best_matching() const {
        Scalar min_cost = INF_np;
        int best_idx = 0;
        
        for (size_t i = 0; i < solutions.size(); ++i) {
            Scalar cost = solutions[i].get_cost();
            if (cost < min_cost) {
                min_cost = cost;
                best_idx = i;
            }
        }

        //curr_omega = solutions[best_idx].omega;

        //std::cout << "Currnet omega: " << curr_omega << std::endl;

        //this->set_current_omega(best_idx);
        
        //make thr best matching
        std::vector<Edge_np> best_matching;
        for (const Server_np& server : solutions[best_idx].servers) {
            if (server.matched_request_idx != -1) {
                best_matching.push_back(Edge_np(server.matched_request_idx, server.matched_request_idx));
            }
        }
        return best_matching;
    }*/
    
    // Get the cost of the best matching
    /*Scalar get_best_cost() const {
        Scalar min_cost = INF_np;
        
        for (const AlgorithmState& state : solutions) {
            min_cost = std::min(min_cost, state.get_cost());
        }
        
        return min_cost;
    }*/

    std::vector<Edge_np> get_matching() const {

        //curr_omega = solutions[best_idx].omega;

        //std::cout << "Currnet omega: " << curr_omega << std::endl;

        //this->set_current_omega(best_idx);
        
        //make thr best matching
        std::vector<Edge_np> best_matching;
        for (int i = 0; i < solutions[0].servers.size(); ++i) {
            if (solutions[0].servers[i].matched_request_idx != -1) {
                best_matching.push_back(Edge_np(i, solutions[0].servers[i].matched_request_idx));
            }
        }
        return best_matching;
    }

    Scalar getTotalcost() const {
        if (inside_opt) {
            return solutions[0].getTotalcost()+optimalAlgo.getTotalCost();
        }
        return solutions[0].getTotalcost();
    }
    
    // Print the current best matching
    /*void print_best_matching() const {
        Scalar min_cost = INF_np;
        int best_idx = 0;
        
        for (size_t i = 0; i < solutions.size(); ++i) {
            Scalar cost = solutions[i].get_cost();
            if (cost < min_cost) {
                min_cost = cost;
                best_idx = i;
            }
        }
        
        std::cout << "Best matching (omega = " << solutions[best_idx].omega << "):" << std::endl;
        std::cout << "Server -> Request mappings:" << std::endl;
        
        for (const Edge_np& e : get_best_matching()) {
            std::cout << "Server " << e.server_idx << " -> Request " << e.request_idx;
            std::cout << " (Distance: " << 
                solutions[best_idx].servers[e.server_idx].location.distance(
                    solutions[best_idx].requests[e.request_idx].location) << ")" << std::endl;
        }
        
        std::cout << "Total matching cost: " << min_cost << std::endl;
    }*/




    void print_matching() const {
        
        std::cout << "Best matching (omega = " << solutions[0].omega << "):" << std::endl;
        std::cout << "Server -> Request mappings:" << std::endl;
        
        for (const Edge_np& e : get_matching()) {
            std::cout << "Server " << e.server_idx << " -> Request " << e.request_idx;
            std::cout << " (Distance: " << 
                solutions[0].servers[e.server_idx].location.distance(
                    solutions[0].requests[e.request_idx].location) << ")" << std::endl;
        }
        
        std::cout << "Total matching cost: " << getTotalcost() << std::endl;
    }
};

#endif // PushRelabel_H