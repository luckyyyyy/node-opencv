#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <sys/resource.h>

using namespace cv;
using namespace std;

void check_memory_usage() {
    while (true) {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        cout << "Memory usage: " << usage.ru_maxrss << " KB" << endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void template_matching(int thread_id) {
    Mat full_img, template_img, result;
    while (true) {
        full_img = imread("full.jpg", IMREAD_COLOR);
        template_img = imread("1.jpg", IMREAD_COLOR);

        if (full_img.empty() || template_img.empty()) {
            cout << "Thread " << thread_id << ": Could not open or find the images!" << endl;
            return;
        }
        matchTemplate(full_img, template_img, result, TM_CCOEFF_NORMED);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

int main() {
    std::thread memory_thread(check_memory_usage);
    vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(template_matching, i);
    }
    for (auto& t : threads) {
        t.join();
    }
    memory_thread.join();

    return 0;
}
