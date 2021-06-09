#ifndef COMMON_H
#define COMMON_H

#include <QLabel>
#include <QMessageBox>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>    // CV_BGR2RGB



//参数1-显示图像的Label，参数2-要显示的Mat
void LabelDisplayMat(QLabel *label, cv::Mat &mat);
cv::Mat LoadImage(QString str = "F:\\lena.jpg");


#endif // COMMON_H
