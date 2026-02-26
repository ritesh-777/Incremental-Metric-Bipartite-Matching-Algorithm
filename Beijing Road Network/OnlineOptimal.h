#ifndef ONLINE_OPTIMAL_H
#define ONLINE_OPTIMAL_H
#include "common_structures.h" // Assume this defines Point with distance()
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <vector>
#include <set>
#include <algorithm>
#include <limits>

using namespace std;
using namespace boost;
// Graph type definitions
typedef adjacency_list<vecS, vecS, directedS,
    property<vertex_distance_t, Scalar, property<vertex_predecessor_t, int>>,
    property<edge_weight_t, Scalar>> Graph;
typedef graph_traits<Graph>::vertex_descriptor Vertex;

class OnlineMetricMatching {
private:
    vector<Point_np> servers;           // Server locations
    vector<Point_np> requests;          // Request locations
    vector<Scalar> y_S;              // Dual weights for servers
    vector<Scalar> y_R;              // Dual weights for requests
    public:
    vector<int> match_M;             // Online matching: request k -> server
    vector<int> match_M_star;        // Offline matching: request k -> server
    vector<int> match_M_star_server; // Offline matching: server -> request
    set<int> free_servers;           // Set of free server indices
    Scalar t;                        
    int n;                           // Number of servers
    DistCache* dist_cache;        // Distance cache pointer

public:
    OnlineMetricMatching(const vector<Point_np>& server_locations, DistCache* cache = nullptr)
        : servers(server_locations), n(server_locations.size()), t(server_locations.size() * server_locations.size() + 1),
          y_S(server_locations.size(), 0), match_M_star_server(server_locations.size(), -1) {
            t = 1;
        for (int i = 0; i < n; ++i) {
            free_servers.insert(i);
        }
        dist_cache = cache;
    }

    OnlineMetricMatching() {}

    void loadServers(const vector<Point_np>& server_locations, DistCache* cache = nullptr) {
        servers = server_locations;
        n = server_locations.size();
        //t = server_locations.size() * server_locations.size() + 1;

        //y_S = (server_locations.size(), 0); match_M_star_server(server_locations.size(), -1) {
        t = 1;
        for (int i = 0; i < n; ++i) {
            y_S.push_back(0);
            match_M_star_server.push_back(-1);
            free_servers.insert(i);
        }
        dist_cache = cache;
    }

    void processRequest(const Point_np& r) {
        
        int k = requests.size();
        requests.push_back(r);
        y_R.push_back(0);
        match_M.push_back(-1);
        match_M_star.push_back(-1);

        int r_idx = n + k; // New request’s vertex index
        int num_vertices = n + k + 1;
        Graph G(num_vertices);

        // Build residual graph
        // Matching edges (server to matched request)
        for (int s = 0; s < n; ++s) {
            if (match_M_star_server[s] != -1) {
                int r_j = match_M_star_server[s];
                add_edge(s, n + r_j, 0, G); // Zero cost for matched edges
            }
        }
        // Non-matching edges (existing requests to servers)
        for (int r_j = 0; r_j < k; ++r_j) {
            for (int s = 0; s < n; ++s) {
                if (match_M_star[r_j] != s) {
                    Scalar cost = t * servers[s].distance_shortest_path(requests[r_j], *dist_cache) - y_S[s] - y_R[r_j];
                    add_edge(n + r_j, s, max(Scalar(0), cost), G);
                }
            }
        }
        // Non-matching edges (new request to servers)
        for (int s = 0; s < n; ++s) {
            Scalar cost = t * servers[s].distance_shortest_path(requests[k], *dist_cache) - y_S[s];
            add_edge(r_idx, s, max(Scalar(0), cost), G);
        }

        // Run Dijkstra’s algorithm
        vector<Scalar> dist(num_vertices, INF_np);
        vector<int> parent(num_vertices, -1);
        dijkstra_shortest_paths(G, r_idx,
            predecessor_map(make_iterator_property_map(parent.begin(), get(vertex_index, G)))
            .distance_map(make_iterator_property_map(dist.begin(), get(vertex_index, G))));

        // Find free server with minimum distance
        int s_star = -1;
        Scalar min_dist = INF_np;
        for (int s : free_servers) {
            if (dist[s] < min_dist - EPS_np) {
                min_dist = dist[s];
                s_star = s;
            }
        }
        Scalar d = min_dist;

        // Update dual weights
        for (int v = 0; v < num_vertices; ++v) {
            if (dist[v] < d - EPS_np) {
                if (v < n) y_S[v] -= (d - dist[v]);       // Server
                else y_R[v - n] += (d - dist[v]);         // Request
            }
        }

        // Reconstruct augmenting path
        vector<int> path;
        int v = s_star;
        while (v != r_idx) {
            path.push_back(v);
            v = parent[v];
        }
        path.push_back(r_idx);
        reverse(path.begin(), path.end());

        // Update offline matching
        for (size_t i = 0; i < path.size() - 1; i += 2) {
            int r_vertex = path[i];
            int s = path[i + 1];
            int r = r_vertex - n;
            match_M_star[r] = s;
            match_M_star_server[s] = r;
        }

        // Second dual update
        for (size_t i = 0; i < path.size(); i += 2) {
            int r_vertex = path[i];
            int r = r_vertex - n;
            int s = match_M_star[r];
            y_R[r] -= (t - 1) * servers[s].distance_shortest_path(requests[r], *dist_cache);
        }

        // Update online matching
        match_M[k] = s_star;
        free_servers.erase(s_star);
    }

    Scalar getTotalCost() const {
        Scalar total = 0;
        for (size_t k = 0; k < requests.size(); ++k) {
            int s = match_M_star[k];
            total += servers[s].distance_shortest_path(requests[k], *dist_cache);
        }
        return total;
    }
};

#endif