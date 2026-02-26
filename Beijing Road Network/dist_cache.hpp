#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <sys/stat.h> // for file size check on POSIX. On Windows you can skip size check or use _stat.

static inline std::string trim(const std::string &s) {
    size_t a = 0;
    size_t b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b-1]))) --b;
    return s.substr(a, b - a);
}

class DistCache {
public:
    int n;
    float INF;
    std::vector<float> dists;
    std::unordered_map<std::string, int> id_to_index;
    std::vector<std::string> nodes_in_order;

    DistCache(const std::string& bin_path,
              const std::string& nodes_path,
              const std::string& meta_path)
    {
        load_meta(meta_path);
        load_nodes(nodes_path);
        load_matrix(bin_path);
    }

    float get_distance_from_cache(const std::string& src,
                                  const std::string& tgt,
                                  bool verbose = false) const
    {
        std::string s = trim(src);
        std::string t = trim(tgt);

        auto it1 = id_to_index.find(s);
        auto it2 = id_to_index.find(t);

        if (it1 == id_to_index.end() || it2 == id_to_index.end()) {
            if (verbose) {
                std::cerr << "One or both nodes not found: '" << s << "' '" << t << "'\n";
                std::cerr << "Available nodes around start: \n";
                for (int k = 0; k < std::min(n, 10); ++k) {
                    std::cerr << "  [" << k << "] '" << nodes_in_order[k] << "'\n";
                }
            }
            return INFINITY;
        }

        int i = it1->second;
        int j = it2->second;

        float val = dists[ static_cast<size_t>(i) * n + j ];
        if (val >= INF) {
            if (verbose) {
                std::cerr << "Distance value is >= INF sentinel. val=" << val << " INF=" << INF << "\n";
            }
            return INFINITY;
        }
        if (verbose) {
            std::cerr << "Found indices: src_idx=" << i << " tgt_idx=" << j << " raw_val=" << val << "\n";
        }
        return val;
    }

private:
    void load_meta_old(const std::string& meta_path) {
        std::ifstream f(meta_path);
        if (!f.is_open()) throw std::runtime_error("Failed to open meta.json");
        std::string line;
        n = -1;
        INF = -1.0f;
        while (std::getline(f, line)) {
            auto posn = line.find("\"n\"");
            if (posn != std::string::npos) {
                auto colon = line.find(":", posn);
                if (colon != std::string::npos) {
                    std::string num = line.substr(colon+1);
                    // remove non-digit chars
                    std::string digits;
                    for (char c: num) if (std::isdigit(static_cast<unsigned char>(c))) digits.push_back(c);
                    if (!digits.empty()) n = std::stoi(digits);
                }
            }
            auto posi = line.find("\"INF\"");
            if (posi != std::string::npos) {
                auto colon = line.find(":", posi);
                if (colon != std::string::npos) {
                    std::string num = line.substr(colon+1);
                    // strip spaces, commas
                    std::string cleaned;
                    for (char c: num) if (!std::isspace(static_cast<unsigned char>(c)) && c!=',') cleaned.push_back(c);
                    try {
                        INF = std::stof(cleaned);
                    } catch(...) { /* ignore parse errors */ }
                }
            }
        }
        if (n <= 0) throw std::runtime_error("Invalid n in meta.json");
        if (!(INF > 0)) throw std::runtime_error("Invalid INF in meta.json");
    }

    // Replace the previous load_meta(...) with this safer implementation.
    void load_meta(const std::string& meta_path) {
        std::ifstream f(meta_path);
        if (!f.is_open())
            throw std::runtime_error("Failed to open meta.json");

        // Read entire file into string
        std::string content;
        {
            std::ostringstream ss;
            ss << f.rdbuf();
            content = ss.str();
        }

        // helper: find value substring after a given key
        auto extract_value_after_key = [&](const std::string &key) -> std::string {
            size_t pos = content.find(key);
            if (pos == std::string::npos) return "";
            size_t colon = content.find(':', pos);
            if (colon == std::string::npos) return "";
            // start after colon
            size_t i = colon + 1;
            // skip spaces
            while (i < content.size() && std::isspace(static_cast<unsigned char>(content[i]))) ++i;
            if (i >= content.size()) return "";

            // If value starts with quote -> string; otherwise read until comma or closing brace
            if (content[i] == '"') {
                ++i;
                size_t start = i;
                while (i < content.size() && content[i] != '"') ++i;
                return content.substr(start, i - start);
            } else {
                size_t start = i;
                // allow signs, digits, decimal point, e/E for scientific notation, and +/- signs
                while (i < content.size() && (std::isdigit(static_cast<unsigned char>(content[i])) 
                    || content[i]=='+' || content[i]=='-' || content[i]=='.' 
                    || content[i]=='e' || content[i]=='E')) {
                    ++i;
                }
                return content.substr(start, i - start);
            }
        };

        // Extract n
        std::string n_str = extract_value_after_key("\"n\"");
        if (n_str.empty()) {
            // also try without quotes (in case JSON formatted differently)
            n_str = extract_value_after_key("n");
        }
        if (n_str.empty()) {
            std::ostringstream oss;
            oss << "Failed to parse 'n' from meta.json; file content:\n" << content;
            throw std::runtime_error(oss.str());
        }

        // Extract INF
        std::string inf_str = extract_value_after_key("\"INF\"");
        if (inf_str.empty()) {
            inf_str = extract_value_after_key("INF");
        }
        if (inf_str.empty()) {
            std::ostringstream oss;
            oss << "Failed to parse 'INF' from meta.json; file content:\n" << content;
            throw std::runtime_error(oss.str());
        }

        // parse safely
        try {
            // use long long for safety, then check range for int
            long long n_ll = std::stoll(n_str);
            if (n_ll <= 0 || n_ll > static_cast<long long>(1e8)) {
                std::ostringstream oss;
                oss << "Parsed 'n' out of reasonable range: " << n_ll;
                throw std::runtime_error(oss.str());
            }
            n = static_cast<int>(n_ll);
        } catch (const std::exception &ex) {
            std::ostringstream oss;
            oss << "Error parsing 'n' value '" << n_str << "': " << ex.what()
                << "\nmeta.json content:\n" << content;
            throw std::runtime_error(oss.str());
        }

        try {
            // stod handles scientific notation
            double inf_d = std::stod(inf_str);
            INF = static_cast<float>(inf_d);
        } catch (const std::exception &ex) {
            std::ostringstream oss;
            oss << "Error parsing 'INF' value '" << inf_str << "': " << ex.what()
                << "\nmeta.json content:\n" << content;
            throw std::runtime_error(oss.str());
        }
    }


    void load_nodes(const std::string& nodes_path) {
        std::ifstream f(nodes_path);
        if (!f.is_open()) throw std::runtime_error("Failed to open nodes.txt");
        std::string line;
        int idx = 0;
        while (std::getline(f, line)) {
            std::string t = trim(line);
            if (t.empty()) continue;
            nodes_in_order.push_back(t);
            id_to_index[t] = idx++;
        }
        if (idx != n) {
            std::ostringstream oss;
            oss << "nodes.txt count (" << idx << ") does not match n (" << n << ")";
            throw std::runtime_error(oss.str());
        }
    }

    void load_matrix(const std::string& bin_path) {
        // check file size (optional but useful)
        struct stat st;
        if (stat(bin_path.c_str(), &st) != 0) {
            throw std::runtime_error("Failed to stat dists.bin");
        }
        long expected = static_cast<long>( (long long)n * n * sizeof(float) );
        if (st.st_size != expected) {
            std::ostringstream oss;
            oss << "dists.bin size mismatch. got=" << st.st_size << " expected=" << expected;
            throw std::runtime_error(oss.str());
        }

        std::ifstream f(bin_path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Failed to open dists.bin");
        dists.resize( static_cast<size_t>(n) * n );
        f.read(reinterpret_cast<char*>(dists.data()), static_cast<std::streamsize>(dists.size() * sizeof(float)));
        if (!f) throw std::runtime_error("Failed reading all bytes from dists.bin");
    }
};
