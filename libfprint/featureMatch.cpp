#include <stdio.h>
#include <opencv2/opencv.hpp>
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
#include <iostream>
#include <glib.h>
#include <featureMatch.hpp>

// this is actually defined in bozorth.h
// guess one could use that one instead and remove the conflicting
// include atanf from opencv
#define MAX_BOZORTH_MINUTIAE 200
struct xyt_struct {
	int nrows;
	int xcol[     MAX_BOZORTH_MINUTIAE ];
	int ycol[     MAX_BOZORTH_MINUTIAE ];
	int thetacol[ MAX_BOZORTH_MINUTIAE ];
        unsigned char *binarized;
        unsigned int   wid;
        unsigned int   hei;
};

using namespace std;
using namespace cv;

void swap(float *p, float *q);
void sort(float a[], int n);
Mat get_section(Mat img, int win, int xc, int yc);
void process_matches_AK(Mat* img1, Mat* img2, vector<KeyPoint> * matched1, 
                        vector<KeyPoint> * matched2);

Mat get_section(Mat img, int win, int xc, int yc)
{
  int xmin, xmax, ymin, ymax;  
  int hei = img.rows;
  int wid = img.cols;
  xmin = xc-win;
  xmax = xc+win;
  ymin = yc-win;
  ymax = yc+win;
  if (xmin < 0)
  {
    xmin = 0;
    xmax = 2*win;
  } else if (xmax > (wid-1))
  {
    xmax = wid-1;
    xmin = wid-2*win-2;
  }
  if (ymin < 0)
  {
    ymin = 0;
    ymax = 2*win;
  } else if (ymax > (hei-1))
  {
    ymax = hei-1;
    ymin = hei-2*win-2;
  }
  Mat temp;
  img(cv::Rect(xmin,ymin,win*2+1,win*2+1)).copyTo(temp);
  return temp;
}

void process_matches_AK(Mat* img1, Mat* img2, vector<KeyPoint> * matched1,
                        vector<KeyPoint> * matched2)
{
  Mat desc1, desc2;
  vector<KeyPoint> kp1, kp2;
  
  // use akaze for feature detection
  Ptr<AKAZE> akaze = AKAZE::create(AKAZE::DESCRIPTOR_MLDB,0,3,0.001f,1,1,KAZE::DIFF_PM_G2);

  akaze->detectAndCompute(*img1, noArray(), kp1, desc1);
  akaze->detectAndCompute(*img2, noArray(), kp2, desc2);
  
  // detect matches in forward direction
  BFMatcher matcher(NORM_HAMMING);
  vector< vector<DMatch> > matches;
  matcher.knnMatch(desc1, desc2, matches, 2);
  int n = (int) matches.size();
  
  g_debug("initial number of matches: %d", n);
  
  // detect matches in reverse direction
  vector< vector<DMatch> > matches_rev;
  matcher.knnMatch(desc2, desc1, matches_rev, 1);

  // first match distance must be better than second*0.77
  // first match distance must be better than 200
  // do the reverse matches match the forward matches?!
  for(int i = 0; i < n; i++) {
    DMatch first = matches[i][0];
    DMatch second = matches[i][1];
    if(first.distance < 0.9 * second.distance && first.distance < 300.0) {
      for(int j = 0; j < (int)matches_rev.size(); j++) {
        DMatch first_rev = matches_rev[j][0];
        if ((first.queryIdx == first_rev.trainIdx) && (first.trainIdx == first_rev.queryIdx)) {
          matched1->push_back(kp1[first.queryIdx]);
          matched2->push_back(kp2[first.trainIdx]);
        }
      }
    }
  }
}

void swap(float *p, float *q) {
   float t;
   
   t=*p; 
   *p=*q; 
   *q=t;
}

void sort(float a[], int n) {
   int i,j;

   for(i = 0; i < n-1; i++) {
      for(j = 0; j < n-i-1; j++) {
         if(a[j] > a[j+1])
            swap(&a[j],&a[j+1]);
      }
   }
}

extern "C" int performMatchCV(struct xyt_struct *xref, struct xyt_struct *xtemp)
{
  Mat img = Mat(xref->hei, xref->wid, CV_8UC1, xref->binarized, Mat::AUTO_STEP);
  Mat templ = Mat(xtemp->hei, xtemp->wid, CV_8UC1, xtemp->binarized, Mat::AUTO_STEP);
  
  vector<KeyPoint> matched1, matched2, matched1c, matched2c;
  
  if (false)
  {
    cout << xtemp->wid << endl;
    cout << xtemp->hei << endl;
    cout << xref->wid << endl;
    cout << xref->hei << endl;
    imshow("template", templ);
    imshow("FULL", img);
    int k = waitKey(0);
  }
  
  if(img.empty() || templ.empty())
  {
    g_debug("Can't read one of the images");
    return 0;
  }
  
  // perform the akaze matching on full fingerprint first
  process_matches_AK(&templ, &img, &matched1, &matched2);
  int n = (int) matched1.size();
  g_debug("number of matches (first run): %d", n);
  
  if (n < 10) {
    g_debug("NO MATCH\nNO MATCH\nNO MATCH\nNO MATCH\nNO MATCH\n");
    return n;
  }
  
  // get cutout from FULL fingerprint to focus on relevant region
  float xm[n] = {0};
  float ym[n] = {0};
  for(int i = 0; i < n; i++) {
    KeyPoint kp = matched2[i];
    xm[i] = kp.pt.x;
    ym[i] = kp.pt.y;
    //g_debug(xm[i]);
    //g_debug(ym[i]);
  }
  sort(xm,n);
  sort(ym,n);
  n = (n+1) / 2 - 1;
  g_debug("median (x): %e", xm[n]);
  g_debug("median (y): %e", ym[n]);
  
  Mat img_cut = get_section(img, 120, (int) xm[n], (int) ym[n]);
  //imshow("FULL", img_cut);
  
  // perform the akaze matching on cutout
  process_matches_AK(&templ, &img_cut, &matched1c, &matched2c);
  int nC = (int) matched1c.size();
  g_debug("number of matches (final run): %d", nC);
  
  if (nC < 15) {
    g_debug("NO MATCH\nNO MATCH\nNO MATCH\nNO MATCH\nNO MATCH\n");
    return nC*0.5;
  }
  
  // create points
  vector<Point2f> p1;
  vector<Point2f> p2;
  for(int i = 0; i < nC; i++) {
    p1.push_back(matched1c[i].pt);
    p2.push_back(matched2c[i].pt);
  }
  
  // create DMatch vector
  vector<DMatch> good_matches;
  for(int i = 0; i < nC; i++) {
    good_matches.push_back(DMatch(i, i, 0));
  }
  
  // determine homography
  Mat mask;
  Mat H = findHomography(p1, p2, RANSAC, 3, mask);
  int nH = 0;
  for(int i = 0; i < mask.rows; i++) {
    if (mask.at<uchar>(0,i) == 1) {
      nH++;
    }
  }
  g_debug("number of matches (after homography): %d", nH);
  if (nH < 10) {
    g_debug("NO MATCH\nNO MATCH\nNO MATCH\nNO MATCH\nNO MATCH\n");
    return nH;
  }
  
  // corner points of template
  std::vector<Point2f> p1c(4);
  std::vector<Point2f> p2c(4);
  p1c[0] = Point2f(0, 0);
  p1c[1] = Point2f( (float)templ.cols, 0 );
  p1c[2] = Point2f( (float)templ.cols, (float)templ.rows );
  p1c[3] = Point2f( 0, (float)templ.rows );
  
  // corner points within image
  perspectiveTransform( p1c, p2c, H);
  
  // draw lines between corners
  line( img_cut, p2c[0], p2c[1], Scalar(0, 255, 0), 4 );
  line( img_cut, p2c[1], p2c[2], Scalar(0, 255, 0), 4 );
  line( img_cut, p2c[2], p2c[3], Scalar(0, 255, 0), 4 );
  line( img_cut, p2c[3], p2c[0], Scalar(0, 255, 0), 4 );
  
  // plot matches
  if (false)
  {
    Mat res;
    drawMatches(templ, matched1c, img_cut, matched2c, good_matches, res,
                Scalar::all(-1), Scalar::all(-1), mask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    imshow("result", res);
  }
  
  // we have a MATCH!
  g_debug("MATCH\nMATCH\nMATCH\nMATCH\nMATCH\n");
  return nH;
}
