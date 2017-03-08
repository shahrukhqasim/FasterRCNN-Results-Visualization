#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iterator>
#include <algorithm>
#include <fstream>
#include <regex>
#include <sstream>
#include "unordered_map"


using namespace std;
using namespace cv;

std::vector<std::string> regexSplit(const std::string &s, std::string rgx_str) {
    std::vector<std::string> elems;
    std::regex rgx(rgx_str);
    std::sregex_token_iterator iter(s.begin(), s.end(), rgx, -1);
    std::sregex_token_iterator end;
    while (iter != end) {
        elems.push_back(*iter);
        ++iter;
    }
    return elems;
}


/**
 * This function computes connected components from an input binary image
 *
 * @param[in]     binary Input binary image with background labeled as 0 and foreground as 1.
 * @param[out]    blobs Output vector of connected components containing all pixels belonging to each component.
 * @param[out]    blob_rects Output vector of connected component bounding boxes.
 */
void conComps(const Mat &binary, vector<vector<Point2i> > &blobs, vector<Rect> &blob_rects) {
    blobs.clear();
    blob_rects.clear();
    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    cv::Mat label_image;
    binary.convertTo(label_image, CV_32FC1); // weird it doesn't support CV_32S!

    int label_count = 2; // starts at 2 because 0,1 are used already

    for (int y = 0; y < binary.rows; y++) {
        for (int x = 0; x < binary.cols; x++) {
            if ((int) label_image.at<float>(y, x) != 1) {
                continue;
            }

            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x, y), cv::Scalar(label_count), &rect, cv::Scalar(0), cv::Scalar(0),
                          4);

            std::vector<cv::Point2i> blob;

            for (int i = rect.y; i < (rect.y + rect.height); i++) {
                for (int j = rect.x; j < (rect.x + rect.width); j++) {
                    if ((int) label_image.at<float>(i, j) != label_count) {
                        continue;
                    }

                    blob.push_back(cv::Point2i(j, i));
                }
            }
            blobs.push_back(blob);

            label_count++;
        }
    }

    for (size_t i = 0; i < blobs.size(); i++) {
        int top = 10000, bottom = -1, left = 10000, right = -1;
        for (size_t j = 0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;
            if (x < left)
                left = x;
            if (x > right)
                right = x;
            if (y < top)
                top = y;
            if (y > bottom)
                bottom = y;
        }
        int w = right - left;
        int h = bottom - top;
        blob_rects.push_back(Rect(left, top, w, h));
    }
}

void binarizeShafait(Mat &gray, Mat &binary, int w, double k) {
    Mat sum, sumsq;
    gray.copyTo(binary);
    int half_width = w >> 1;
    integral(gray, sum, sumsq, CV_64F);
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            int x_0 = (i > half_width) ? i - half_width : 0;
            int y_0 = (j > half_width) ? j - half_width : 0;
            int x_1 = (i + half_width >= gray.rows) ? gray.rows - 1 : i + half_width;
            int y_1 = (j + half_width >= gray.cols) ? gray.cols - 1 : j + half_width;
            double area = (x_1 - x_0) * (y_1 - y_0);
            double mean = (sum.at<double>(x_0, y_0) + sum.at<double>(x_1, y_1) - sum.at<double>(x_0, y_1) -
                           sum.at<double>(x_1, y_0)) / area;
            double sq_mean = (sumsq.at<double>(x_0, y_0) + sumsq.at<double>(x_1, y_1) - sumsq.at<double>(x_0, y_1) -
                              sumsq.at<double>(x_1, y_0)) / area;
            double stdev = sqrt(sq_mean - (mean * mean));
            double threshold = mean * (1 + k * ((stdev / 128) - 1));
            if (gray.at<uchar>(i, j) > threshold)
                binary.at<uchar>(i, j) = 255;
            else
                binary.at<uchar>(i, j) = 0;
        }
    }
}

/**
 * First argument = images directory
 * Second argument = GT directory (should contains ids.txt)
 * Third argument = Faster R-CNN output file path
 * Fourth argument = Output directory path
 *
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char *argv[]) {
    if (argc != 5) {
        cout << "Error in arguments, breaking!\n";
        cout << "Figure is " << argc << endl;
        return 0;
    }

    string imagesDir = argv[1];
    string gtDir = argv[2];
    string fasterFilePath = argv[3];
    string outputDir = argv[4];

    if (gtDir[gtDir.length() - 1] != '/')
        gtDir += '/';
    if (imagesDir[imagesDir.length() - 1] != '/')
        imagesDir += '/';
    if (outputDir[outputDir.length() - 1] != '/')
        outputDir += '/';

    ifstream gtIdsFile(gtDir + "ids.txt");
    vector<string> groundTruthIds;
    copy(istream_iterator<string>(gtIdsFile),
         istream_iterator<string>(),
         back_inserter(groundTruthIds));


    ifstream fasterOutputFile(fasterFilePath);
    vector<string> fasterAll;
    std::string s;
    while (std::getline(fasterOutputFile, s)) {
        fasterAll.push_back(s);
    }

    unordered_map<string, vector<Rect>> fasterOutputRectangles; // "id"=>{box1, box2, box3, ...}
    for (auto i:fasterAll) {
        vector<string> boundingBoxCoordinatesStrings = regexSplit(i, "\\s+");
        int x1, y1, x2, y2;
        x1 = stof(boundingBoxCoordinatesStrings[2]);
        y1 = stof(boundingBoxCoordinatesStrings[3]);
        x2 = stof(boundingBoxCoordinatesStrings[4]);
        y2 = stof(boundingBoxCoordinatesStrings[5]);

        vector<Rect> r;
        if (fasterOutputRectangles.find(boundingBoxCoordinatesStrings[0]) != fasterOutputRectangles.end()) {
            r = fasterOutputRectangles[boundingBoxCoordinatesStrings[0]];
        }
        r.push_back(Rect(x1, y1, x2 - x1, y2 - y1));
        fasterOutputRectangles[boundingBoxCoordinatesStrings[0]] = r;
    }
    int correct=0,partial=0;
    int missed=0;
    int falsePositive=0;
    int overSegmented=0;
    float AreaCorrect,AreaMissed,AreaFalsePositive,precision,AreaOverSegmented;
    vector<Rect>rcnnTrimmedBoundingBoxes;
    vector<Rect>groundTruthBoundingBoxes;
    float averagePrecision;
    float maxThresh=0.9;
    float minThresh=0.1;

    for (auto i:groundTruthIds) {
        ifstream ifs(gtDir + i + ".txt");
        string gtContent((istreambuf_iterator<char>(ifs)),
                         (istreambuf_iterator<char>()));
        vector<string> boundingBoxesStrings = regexSplit(gtContent, "\n");
        vector<Rect> groundTruthBoundingBoxes;
        cout << "Reading " << imagesDir + i + ".png" << endl;
        Mat image = imread(imagesDir + i + ".png", 1);
        Mat grayScale;
        cvtColor(image, grayScale, CV_BGR2GRAY);
        Mat binaryImage;
        binarizeShafait(grayScale, binaryImage, 50, 0.30);

        binaryImage /= 255;
        binaryImage = 1 - binaryImage;

        vector<vector<Point2i> > blobs;
        vector<Rect> conCompsRects;
        conComps(binaryImage, blobs, conCompsRects);

        for (auto j:boundingBoxesStrings) {
            vector<string> boundingBoxCoordinates = regexSplit(j, "[,]");
            int x1, y1, x2, y2;
            x1 = stoi(boundingBoxCoordinates[0]);
            y1 = stoi(boundingBoxCoordinates[1]);
            x2 = stoi(boundingBoxCoordinates[2]);
            y2 = stoi(boundingBoxCoordinates[3]);
            rectangle(image, Rect(x1, y1, x2 - x1, y2 - y1), Scalar(255, 0, 0), 3);
            groundTruthBoundingBoxes.push_back(Rect(x1, y1, x2 - x1, y2 - y1));
        }

        vector<Rect> rcnnBoundingBoxes;
        if (fasterOutputRectangles.find(i) != fasterOutputRectangles.end()) {

            rcnnBoundingBoxes = fasterOutputRectangles[i];

        }

        // rcnnBoundingBoxes will contain rcnn output boxes
        // groundTruthBoundingBoxes will contain GT bounding boxes

        vector<Rect> rcnnTrimmedBoundingBoxes;
        for (auto b:rcnnBoundingBoxes) {
            vector<Rect> innerRects;
            for (auto j:conCompsRects) {
                if ((j & b).area() > 0) {
                    innerRects.push_back(j);
                }
            }
            // innerRects will contain all the boxes which are inside b
            Rect trimmed;
            bool isSet = false;
            for (auto j:innerRects) {
                if (isSet) {
                    trimmed = (j & b) | trimmed;
                } else {
                    trimmed = j & b;
                    isSet = true;
                };
            }
            rcnnTrimmedBoundingBoxes.push_back(trimmed);
        }

        //rcnnTrimmedBoundingBoxes contains trimmed boundix boxes of faster rcnn

        for (auto j:rcnnTrimmedBoundingBoxes) {
            rectangle(image, j, Scalar(0, 0, 255), 3);
        }
        //Correct & Partial
        for(auto i:groundTruthBoundingBoxes){
            for(auto j:rcnnTrimmedBoundingBoxes ) {
                AreaCorrect = 2 * abs((i & j).area()) / (float)abs((i | j).area());
                if (AreaCorrect >= maxThresh)
                    correct += 1;
                else if (AreaCorrect<minThresh && AreaCorrect>maxThresh)
                    partial += 1;


            }
        }

        //Over-Segmented
        for(auto i:groundTruthBoundingBoxes){
            for(auto j:rcnnTrimmedBoundingBoxes ) {

                AreaOverSegmented=abs((i&j).area())/(float)abs(i.area());
              // if(AreaOverSegmented>minThresh && AreaOverSegmented<maxThresh)
                 if(AreaOverSegmented>0.9)
                    overSegmented+=1;

            }
        }

        int sizeRcnn=rcnnTrimmedBoundingBoxes.size();
        //Missed
        for(auto i:groundTruthBoundingBoxes){
            for(auto j: rcnnTrimmedBoundingBoxes) {
                AreaMissed=abs((i&j).area())/(float)abs(i.area());
                if (AreaMissed<=minThresh)
                    missed+=1;

            }
        }
        //False Positives
        for(auto i:rcnnTrimmedBoundingBoxes){
            for(auto j:groundTruthBoundingBoxes ) {
               // AreaFalsePositive=(i&j).area()/(float)i.area();
                AreaFalsePositive=(float)abs((i&j).area());
                if (AreaFalsePositive<=minThresh)
                {
                    falsePositive+=1;
                }

            }
        }
        //Precision
        for(auto i:groundTruthBoundingBoxes){
            for(auto j: rcnnTrimmedBoundingBoxes) {
                precision = (abs((i & j).area()) / (float)abs( (i.area())));
                averagePrecision=precision/(float)sizeRcnn;

            }
        }

        imwrite(outputDir + i + ".png", image);
    }

    cout << "Total correct are " << correct << endl;
    cout << "Total over-segmented are " << overSegmented << endl;
    cout <<"Total false positives are "<<falsePositive<<endl;
    cout<<"Total missed are "<<missed<<endl;
    cout<<"Total partial are "<<partial<<endl;
    cout<< "MAP is " << averagePrecision << endl;
    return 0;
}