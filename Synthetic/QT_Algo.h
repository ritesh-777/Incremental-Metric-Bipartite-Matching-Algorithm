#ifndef QT_ALGO_H
#define QT_ALGO_H
#include <iostream>
#include <memory>
#include <map>
#include <cmath>
#include <vector>
#include <optional>
#include "common_structures.h"



// Axis-aligned bounding box
template<typename T>
struct AABB {
    T xmin, ymin, xmax, ymax;
    bool contains(const Point_np &p) const {
        return p.x >= xmin && p.x <= xmax && p.y >= ymin && p.y <= ymax;
    }
    bool intersects(const AABB &o) const {
        return !(o.xmin > xmax || o.xmax < xmin || o.ymin > ymax || o.ymax < ymin);
    }
};

struct Node {
    AABB<double> box;
    bool isLeaf = true;
    std::optional<Point_np> server;      // holds one server if leaf
    std::vector<const Point_np*> subtree_servers;    // holds all servers in subtree
    std::unique_ptr<Node> children[4];
    Node *parent = nullptr;
    int serverCount = 0;               // subtree total
    int discrepancy = 0;               // number of servers matched to requests in this subtree
    int tin = 0;
    int tout = 0;

    Node(const AABB<double>& b, Node* p=nullptr)
      : box(b), parent(p) {}

    bool contains(const Point_np &p) const {
        if(!isLeaf){
            return box.contains(p);
        }
        else{
            if(server){
                return server->x == p.x && server->y == p.y;
            }
            else{
                return box.contains(p);
            }
        }
    }
};

class Quadtree {
public:
    std::unique_ptr<std::map<Point_np, std::unique_ptr<std::map<Point_np,bool>>>> past_matchings;
    Quadtree(const AABB<double>& bounds)
      : root_(std::make_unique<Node>(bounds)){
        timer_ = 1;
        dfsStamp(root_.get());
        past_matchings = std::make_unique<decltype(past_matchings)::element_type>();
      }

    // Insert a server point
    void insertServer(const Point_np &p) {
        (*past_matchings)[p] = std::make_unique<std::map<Point_np,bool>>();
        insertPoint(root_.get(), p);

    }

    // Match a request to a free server
    std::optional<Point_np> matchRequest(const Point_np &req) {
        return matchRequestInternal(req);
    }

    double getTotalcost() const {
        double cost = 0;
        for (auto &pr : matched_) {
            Point_np s = *pr.first, r = pr.second;
            cost += s.distance(r);
        }
        return cost;
    }

    void print() const {
        printNode(root_.get(), 0);
    }

    // print matching
    void printMatching() const {
        for (const auto &pr : matched_) {
            std::cout << "Server: (" << (*pr.first).x << ", " << (*pr.first).y
                      << ") -> Request: (" << pr.second.x << ", " << pr.second.y
                      << ")" << std::endl;
        }
    }

private:
    //bool is_server = false;
    std::unique_ptr<Node> root_;
    std::map<const Point_np*, Point_np> matched_;    // server -> request
    std::map<Point_np, Node*> lastLCA_;       // request -> leaf node
    int timer_ = 0;

    void printNode(Node* n, int d) const {
        if (!n) return;
        std::cout << std::string(d*2,' ')
                  << "Node[" << n->box.xmin << "," << n->box.ymin
                  << ";" << n->box.xmax << "," << n->box.ymax
                  << "] count=" << n->serverCount
                  << " disc=" << n->discrepancy;
        if (n->isLeaf && n->server)
            std::cout << " srv=(" << n->server->x << "," << n->server->y << ")";
        std::cout << std::endl;
        if (!n->isLeaf) {
            for (auto &c: n->children)
                printNode(c.get(), d+1);
        }
    }

    void dfsStamp(Node* n) {
        if (!n) return;
        n->tin = timer_++;
        if (!n->isLeaf) {
            for (auto &c : n->children) {
                dfsStamp(c.get());
            }
        }
        n->tout = timer_++;
    }

    bool isInSubtree(Node* subRoot, Node* x) {
        return subRoot->tin <= x->tin && x->tout <= subRoot->tout;
    }

    void subdivide(Node* n) {
        double xmid = (n->box.xmin + n->box.xmax)/2;
        double ymid = (n->box.ymin + n->box.ymax)/2;
        AABB<double> bs[4] = {
            {n->box.xmin, n->box.ymin, xmid, ymid},
            {xmid, n->box.ymin, n->box.xmax, ymid},
            {n->box.xmin, ymid, xmid, n->box.ymax},
            {xmid, ymid, n->box.xmax, n->box.ymax}
        };
        for (int i=0;i<4;i++) {
            n->children[i] = std::make_unique<Node>(bs[i], n);
        }
        if (n->server) {
            for (auto &c: n->children) {
                if (c->box.contains(*n->server)) {
                    c->server = n->server;
                    for (auto &s : n->subtree_servers) {
                        c->serverCount++;
                        c->discrepancy++;
                        c->subtree_servers.push_back(s);
                    }
                    break;
                }
            }
            n->server.reset();
        }
        n->isLeaf = false;
        timer_ = 1;
        dfsStamp(root_.get());
    }

    void insertPoint(Node* n, const Point_np &p) {
        if (!n->box.contains(p)) return;
        if (n->isLeaf) {
            if (!n->server) {
                n->serverCount++;
                n->discrepancy++;
                n->subtree_servers.push_back(&p);
                n->server = p;
            } else if (n->server.value().coords[0]==p.coords[0] && n->server.value().coords[1]==p.coords[1]) { 
                n->serverCount++;
                n->discrepancy++;
                n->subtree_servers.push_back(&p);
            } else {
                subdivide(n);
                insertPoint(n, p);
            }
        } else {
            n->serverCount++;
            n->discrepancy++;
            n->subtree_servers.push_back(&p);
            for (auto &c: n->children)
                if (c->box.contains(p)) {
                    insertPoint(c.get(), p);
                }
        }
    }

    Node* findLeaf(Node* n, const Point_np &p) {
        if (n->isLeaf) return n;
        for (auto &c: n->children){
            if (c->box.contains(p)) return findLeaf(c.get(), p);}
        return n;
    }

    Node* findFirstAncestorWithServer(Node* leaf) {
        Node* cur = leaf;
        while (cur && cur->serverCount == 0) cur = cur->parent;
        return cur;
    }


    std::optional<Point_np> matchRequestInternal(const Point_np &req) {
        Node* leaf = findLeaf(root_.get(), req);
        Node* anc = findFirstAncestorWithServer(leaf);
        while (anc) {
            // 1. Free server available?
            for (auto &pt : anc->subtree_servers) {
                bool free = matched_.count(pt) == false;
                if (free) {
                    // match
                    matched_[pt] = req;
                    //lca of pt,req
                    updateDiscrepancies(anc, -1, nullptr);
                    lastLCA_[req] = anc;
                    //std::cin.get();
                    return (*pt);
                }
            }
            
            // 2. Rematch using discrepancy
            if (anc->discrepancy > 0) {
                for (auto &pt : anc->subtree_servers) {
                    auto it = matched_.find(pt);
                    if (it != matched_.end()) {
                        Point_np oldReq = it->second;
                        // check oldReq outside anc subtree
                        Node* oldLeaf = lastLCA_[oldReq];
                        if (!isInSubtree(anc, oldLeaf)) {
                            matched_[pt] = req;
                            lastLCA_[req] = anc;
                            updateDiscrepancies(anc, -1, oldLeaf);
                            // recurse for displaced req
                            matchRequestInternal(oldReq);
                            return *pt;
                        }
                    }
                }
            }
            // move up
            anc = anc->parent;
        }
        return std::nullopt;
    }

    void updateDiscrepancies(Node* start, int delta, Node* lca, int i = 0) {
        Node* leaf = start;
        if(leaf == lca) {
            leaf->discrepancy += delta;
            return;
        }
        while (leaf != nullptr && leaf != lca) {
            leaf->discrepancy += delta;
            leaf = leaf->parent;
        }
    }
};

#endif // QT_ALGO_H
