#include "ffmpegcv.hpp"
#include "infer.h"
#include <chrono>
#include <iostream>

int process_frame(YoloDetector &detector, uint8_t* frame, int width, int height) {
    cv::Mat img(height, width, CV_8UC3, frame); // 直接映射，不复制数据

    std::vector<Detection> res = detector.inference(img);
    YoloDetector::draw_image(img, res);
    return 0;
}

int main() {
    // Initialize video capture
    ffmpegcv::VideoCaptureNV cap("input.mp4",    "bgr24",
        {0,0,0, 0},  // crop_xywh, {0,0,0,0} for no crop
        {640, 640}   );
    auto frame_size = {640, 640};
    
    // Initialize video writer (H.264 codec)
    ffmpegcv::VideoWriterNV writer("output.mp4", "hevc", cap.fps, frame_size);
    
    // Create YoloDetector and load engine plan
    std::string trtFile = "./best.plan";
    YoloDetector detector(trtFile);
    
    // Frame buffer (BGR format)
    uint8_t* frame = new uint8_t[640 * 640 * 3];
    
    // Processing loop
    while (cap.read(frame)) {
        // Process the current frame
        process_frame(detector, frame, 640, 640);
        
        // Write the processed frame to the output video
        writer.write(frame);
    }
    
    // Cleanup
    delete[] frame;
    cap.release();
    writer.release();
    
    return 0;
}
