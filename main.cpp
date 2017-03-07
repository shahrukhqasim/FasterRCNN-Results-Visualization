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
        cout<<"Figure is "<<argc<<endl;
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

    unordered_map<string,vector<Rect>>fasterOutputRectangles; // "id"=>{box1, box2, box3, ...}
    for(auto i:fasterAll) {
        vector<string>boundingBoxCoordinatesStrings=regexSplit(i,"\\s+");
        int x1,y1,x2,y2;
        x1=stof(boundingBoxCoordinatesStrings[2]);
        y1=stof(boundingBoxCoordinatesStrings[3]);
        x2=stof(boundingBoxCoordinatesStrings[4]);
        y2=stof(boundingBoxCoordinatesStrings[5]);

        vector<Rect>r;
        if(fasterOutputRectangles.find(boundingBoxCoordinatesStrings[0]) != fasterOutputRectangles.end()) {
            r=fasterOutputRectangles[boundingBoxCoordinatesStrings[0]];
        }
        r.push_back(Rect(x1,y1,x2-x1,y2-y1));
        fasterOutputRectangles[boundingBoxCoordinatesStrings[0]]=r;
    }


    for (auto i:groundTruthIds) {
        ifstream ifs(gtDir+i+".txt");
        string gtContent( (istreambuf_iterator<char>(ifs) ),
                             (istreambuf_iterator<char>()    ) );
        vector<string>boundingBoxesStrings=regexSplit(gtContent,"\n");
        vector<Rect>boundingBoxes;
        cout<<"Reading "<<imagesDir+i+".png"<<endl;
        Mat image=imread(imagesDir+i+".png",1);
        for(auto j:boundingBoxesStrings) {
            vector<string>boundingBoxCoordinates=regexSplit(j,"[,]");
            int x1,y1,x2,y2;
            x1=stoi(boundingBoxCoordinates[0]);
            y1=stoi(boundingBoxCoordinates[1]);
            x2=stoi(boundingBoxCoordinates[2]);
            y2=stoi(boundingBoxCoordinates[3]);
            rectangle(image,Rect(x1,y1,x2-x1,y2-y1),Scalar(255,0,0),3);
        }

        vector<Rect>fasterRcnnOutputRectangles;
        if(fasterOutputRectangles.find(i) != fasterOutputRectangles.end()) {
            fasterRcnnOutputRectangles=fasterOutputRectangles[i];
        }
        for(auto j:fasterRcnnOutputRectangles) {
            rectangle(image,j,Scalar(0,255,0),3);

        }

        imwrite(outputDir+i+".png",image);
    }


    return 0;
}