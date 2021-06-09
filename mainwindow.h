#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>

#include "common.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void OpenFile();

private:
    Ui::MainWindow *ui;

    void Recognition();

    cv::Mat Gaussian(cv::Mat& src);
    cv::Mat Grayscale(cv::Mat& src);
    cv::Mat Sobel(cv::Mat& src);
    cv::Mat TwoValued(cv::Mat& src);
    cv::Mat Close(cv::Mat& src);
    cv::Mat Contour(cv::Mat& src);

    // 判定面积是否在规定范围以内
    bool VerifySizes(cv::RotatedRect minArea);



    // 图像校正
    void ImageCorrection_Contour();
    void ImageCorrection_HoughLine();



private:
    cv::Mat m_RawImage;
    cv::Mat m_MidImage;


};
#endif // MAINWINDOW_H
