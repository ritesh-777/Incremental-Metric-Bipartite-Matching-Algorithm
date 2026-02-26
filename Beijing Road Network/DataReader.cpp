#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <random> 
#include <algorithm> 
#include "common_structures.h"

int MAX_RECORDS = 2600000;

class DataSet {
    private :
        std::string filename;
        int number_of_points;
        int dim;

    public :
        DataSet(const std::string& f, int n, int dimension) {
            filename = f;
            std::ifstream file(filename);

            if (!file.is_open()) {
                std::cerr << "Could not open the file: " << filename << std::endl;
            }
            file.close();

            number_of_points = n;
            dim = dimension;
        }

        std::vector<Point_np> server() {
            std::ifstream file(filename);
            if (number_of_points > MAX_RECORDS) {
                std::cerr << "Start index and end index mismatch" << std::endl;
            }
            std::string line;
            std::vector<Point_np> servers;
            std::vector<Scalar> coords(2);
            int line_counter = 0;

            std::vector<int> random_indexes = gen_rand_points();

            //std::cout<<random_indexes.size();

            while (std::getline(file, line)) {// && line_counter <= end_index) {
                //std::cin.get();
                if (random_indexes[0]==line_counter) {//std::find(random_indexes.begin(), random_indexes.end(), line_counter) != random_indexes.end()) {//line_counter >= start_index){
                    random_indexes.erase(random_indexes.begin());
                    std::stringstream ss(line);
                    std::string value;
                    int word_counter = 0;
                    while (std::getline(ss, value, ',')) {
                        if (word_counter == 0 || word_counter == 1) {
                            try {
                                if (word_counter == 0) {
                                    coords[1] = std::stod(value);
                                }
                                else {
                                    coords[0] = std::stod(value);
                                }
                                //Scalar floatValue = std::stod(value);
                                //std::cout << std::setprecision(20);
                                //std::cout << floatValue << " ";
                            } catch (const std::invalid_argument& e) {
                                std::cerr << "Invalid location value: " << value << std::endl;
                            } catch (const std::out_of_range& e) {
                                std::cerr << "value is out of range: " << value << std::endl;
                            }
                        }
                        /*else {
                            try {
                                Scalar floatValue = std::stof(value);
                                std::cout << floatValue << " ";
                            } catch (const std::invalid_argument& e) {
                                std::cerr << "Invalid float value: " << value << std::endl;
                            } catch (const std::out_of_range& e) {
                                std::cerr << "Float value out of range: " << value << std::endl;
                            }
                        }*/
                        word_counter++;
                    }
                    //std::cout << std::endl;
                    servers.push_back(Point_np(coords));
                }
                line_counter++;
            }
            file.close();
            return servers;
        }


        std::vector<Point_np> request() {
            std::ifstream file(filename);
            if (number_of_points > MAX_RECORDS) {
                std::cerr << "Start index and end index mismatch" << std::endl;
            }
            std::string line;
            std::vector<Point_np> requests;
            std::vector<Scalar> coords(2);
            int line_counter = 0;

            std::vector<int> random_indexes = gen_rand_points();

            while (std::getline(file, line)) {// && line_counter <= end_index) {
                //std::cin.get();
                if (random_indexes[0]==line_counter) {//std::find(random_indexes.begin(), random_indexes.end(), line_counter) != random_indexes.end()) {//line_counter >= start_index){
                    random_indexes.erase(random_indexes.begin());
                    std::stringstream ss(line);
                    std::string value;
                    int word_counter = 0;
                    while (std::getline(ss, value, ',')) {
                        if (word_counter == 2 || word_counter == 3) {
                            try {
                                if (word_counter == 2) {
                                    coords[1] = std::stod(value);
                                }
                                else {
                                    coords[0] = std::stod(value);
                                }
                                //Scalar floatValue = std::stod(value);
                                //std::cout << std::setprecision(20);
                                //std::cout << floatValue << " ";
                            } catch (const std::invalid_argument& e) {
                                std::cerr << "Invalid location value: " << value << std::endl;
                            } catch (const std::out_of_range& e) {
                                std::cerr << "value is out of range: " << value << std::endl;
                            }
                        }
                        /*else {
                            try {
                                Scalar floatValue = std::stof(value);
                                std::cout << floatValue << " ";
                            } catch (const std::invalid_argument& e) {
                                std::cerr << "Invalid float value: " << value << std::endl;
                            } catch (const std::out_of_range& e) {
                                std::cerr << "Float value out of range: " << value << std::endl;
                            }
                        }*/
                        word_counter++;
                    }
                    //std::cout << std::endl;
                    requests.push_back(Point_np(coords));
                }
                line_counter++;
            }
            file.close();
            return requests;
        }

        

        std::vector<int> gen_rand_points(){
            // Create a vector with numbers from min to max 
            std::vector<int> numbers; 
            std::vector<int> random_numbers;
            for (int i = 1; i < MAX_RECORDS; ++i) { 
                numbers.push_back(i); 
            } 
 
            // Shuffle the numbers to randomize their order 
            std::random_device rd;  // Obtain a random number from hardware 
            std::mt19937 eng(rd()); // Seed the generator 
            std::shuffle(numbers.begin(), numbers.end(), eng); 
 
            // Print the numbers in random order without repetition 
            for (int i=0; i<number_of_points; ++i) { 
                random_numbers.push_back(numbers[i]);
                //std::cout << numbers[i] << " "; 
            } 
            std::sort(random_numbers.begin(), random_numbers.end());
            //std::cout << std::endl; 

            return random_numbers;

        }



        std::vector<Point_np> server_static() {
            std::ifstream file(filename);
            if (number_of_points > MAX_RECORDS) {
                std::cerr << "Start index and end index mismatch" << std::endl;
            }
            std::string line;
            std::vector<Point_np> servers;
            std::vector<Scalar> coords(dim);
            int line_counter = 0;

            //std::vector<int> random_indexes = gen_rand_points();

            //std::cout<<random_indexes.size();

            while (std::getline(file, line)) {// && line_counter <= end_index) {
                //std::cin.get();
                if (line_counter != 0) {//std::find(random_indexes.begin(), random_indexes.end(), line_counter) != random_indexes.end()) {//line_counter >= start_index){
                    //random_indexes.erase(random_indexes.begin());
                    std::stringstream ss(line);
                    std::string value;
                    int word_counter = 0;
                    while (std::getline(ss, value, ',')) {
                        if (word_counter < dim) {//if (word_counter == 0 || word_counter == 1) {
                            try {
                                coords[word_counter] = std::stod(value);
                                /*if (word_counter == 0) {
                                    coords[0] = std::stod(value);
                                }
                                else {
                                    coords[1] = std::stod(value);
                                }*/
                                //Scalar floatValue = std::stod(value);
                                //std::cout << std::setprecision(20);
                                //std::cout << floatValue << " ";
                            } catch (const std::invalid_argument& e) {
                                std::cerr << "Invalid location value: " << value << std::endl;
                            } catch (const std::out_of_range& e) {
                                std::cerr << "value is out of range: " << value << std::endl;
                            }
                        }
                        /*else {
                            try {
                                Scalar floatValue = std::stof(value);
                                std::cout << floatValue << " ";
                            } catch (const std::invalid_argument& e) {
                                std::cerr << "Invalid float value: " << value << std::endl;
                            } catch (const std::out_of_range& e) {
                                std::cerr << "Float value out of range: " << value << std::endl;
                            }
                        }*/
                        word_counter++;
                    }
                    //std::cout << std::endl;
                    servers.push_back(Point_np(coords));
                }
                line_counter++;
            }
            file.close();
            return servers;
        }


        std::vector<Point_np> request_static() {
            std::ifstream file(filename);
            if (number_of_points > MAX_RECORDS) {
                std::cerr << "Start index and end index mismatch" << std::endl;
            }
            std::string line;
            std::vector<Point_np> requests;
            std::vector<Scalar> coords(dim);
            int line_counter = 0;

            //std::vector<int> random_indexes = gen_rand_points();

            while (std::getline(file, line)) {// && line_counter <= end_index) {
                //std::cin.get();
                if (line_counter != 0) {//std::find(random_indexes.begin(), random_indexes.end(), line_counter) != random_indexes.end()) {//line_counter >= start_index){
                    //random_indexes.erase(random_indexes.begin());
                    std::stringstream ss(line);
                    std::string value;
                    int word_counter = 0;
                    int idx = 0;
                    while (std::getline(ss, value, ',')) {
                        if (word_counter >= dim && word_counter < 2*dim) { //if (word_counter == 2 || word_counter == 3) {
                            try {
                                coords[idx++] = std::stod(value);
                                /*if (word_counter == 2) {
                                    coords[0] = std::stod(value);
                                }
                                else {
                                    coords[1] = std::stod(value);
                                }*/
                                //Scalar floatValue = std::stod(value);
                                //std::cout << std::setprecision(20);
                                //std::cout << floatValue << " ";
                            } catch (const std::invalid_argument& e) {
                                std::cerr << "Invalid location value: " << value << std::endl;
                            } catch (const std::out_of_range& e) {
                                std::cerr << "value is out of range: " << value << std::endl;
                            }
                        }
                        /*else {
                            try {
                                Scalar floatValue = std::stof(value);
                                std::cout << floatValue << " ";
                            } catch (const std::invalid_argument& e) {
                                std::cerr << "Invalid float value: " << value << std::endl;
                            } catch (const std::out_of_range& e) {
                                std::cerr << "Float value out of range: " << value << std::endl;
                            }
                        }*/
                        word_counter++;
                    }
                    //std::cout << std::endl;
                    requests.push_back(Point_np(coords));
                }
                line_counter++;
            }
            file.close();
            return requests;
        }



};

/*int main() {

    DataSet d("genetrated_locations.csv");
    d.server(1, 10000);
    std::cout<<std::endl;
    d.request(1, 10000);
    /*std::string filename = "genetrated_locations.csv";
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filename << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
        //std::cin.get();
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            try {
                Scalar floatValue = std::stod(value);
                std::cout << floatValue << " ";
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid float value: " << value << std::endl;
            } catch (const std::out_of_range& e) {
                 std::cerr << "Float value out of range: " << value << std::endl;
            }
        }
        std::cout << std::endl;
    }

    file.close();
    return 0;
}*/