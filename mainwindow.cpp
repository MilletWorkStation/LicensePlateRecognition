#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QAction>
#include <QFileDialog>


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);


    connect(ui->actionopen, &QAction::triggered, this, &MainWindow::OpenFile);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::OpenFile()
{
    QString strFile = QFileDialog::getOpenFileName(this, "打开文件");
    if(strFile == "")
        return;

    m_RawImage = cv::imread(strFile.toLocal8Bit().data());
    m_MidImage = m_RawImage.clone();

    LabelDisplayMat(ui->lalbelRawImage, m_RawImage);

    Recognition();
}
/*
 * 主要流程
 * 1、校正图片
 * 2、分割出车牌区域并修改尺寸
 *  高斯滤波
 * 3、车牌二值化处理
 * 4、对二值图像进行形态学处理并取反
 * 5、去除字符外的干扰信息
 * 6、形态学处理并提取字符
 * 7、分割字符并保存
 * 8、字符识别输出结果
*/
void MainWindow::Recognition()
{
    ImageCorrection_Contour();

}

cv::Mat MainWindow::Gaussian(cv::Mat& src)
{
    cv::Mat dst;
    cv::GaussianBlur(src, dst, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    return dst;
}
cv::Mat MainWindow::Grayscale(cv::Mat& src)
{
    cv::Mat dst;
    cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);
    return dst;
}
cv::Mat MainWindow::Sobel(cv::Mat& src)
{
    cv::Mat dst;
    cv::Sobel(src, dst, -1, 1, 0, 3);
    return dst;

}
cv::Mat MainWindow::TwoValued(cv::Mat& src)
{
    cv::Mat dst;
    //cv::threshold(src, dst, 0, 255, cv::THRESH_OTSU + cv::THRESH_BINARY);
    cv::threshold(src, dst, 93, 255, cv::THRESH_BINARY);
    return dst;

}
cv::Mat MainWindow::Close(cv::Mat& src)
{
    cv::Mat dst;
    cv::Mat kenal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(23, 5));
    cv::morphologyEx(src, dst, cv::MORPH_CLOSE , kenal);
    cv::morphologyEx(src, dst, cv::MORPH_CLOSE , kenal);

    cv::morphologyEx(src, dst, cv::MORPH_DILATE , kenal);
    cv::morphologyEx(src, dst, cv::MORPH_DILATE , kenal);
    cv::morphologyEx(src, dst, cv::MORPH_DILATE , kenal);

    return dst;
}

cv::RotatedRect MainWindow::Contour(cv::Mat& src)
{
    std::vector<std::vector<cv::Point> > vContours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(src, vContours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    cv::Scalar color = cv::Scalar(0, 0, 255);

    double dMaxArea = -1.0, dTempArea = -1.0;
    int nPerfectIndex = -1;
    for (size_t i = 0; i < vContours.size(); i++)
    {
        cv::RotatedRect rect = cv::minAreaRect(vContours[i]);

        // 宽高比在范围内
        if(VerifySizes(rect))
        {
            //drawContours(m_MidImage, vContours, (int)i, color, 1, cv::LINE_AA, hierarchy, 0);
            //cv::putText(m_MidImage, QString("%1").arg(rect.angle).toLocal8Bit().data(), vContours[i][0], 0, 0.5, cv::Scalar(0, 255, 0));
//            cv::putText(m_MidImage, QString("%1:%2").arg(i).arg(rect.angle).toLocal8Bit().data(), rect.center, 0, 0.5, cv::Scalar(0, 255, 0));

           // 面积最大的存储
            dTempArea = cv::contourArea(vContours[i]);
            if( dTempArea > dMaxArea )
            {
                dMaxArea = dTempArea;
                nPerfectIndex = i;
            }
        }
    }

    if( nPerfectIndex >= 0 )
        return cv::minAreaRect(vContours[nPerfectIndex]);

    return cv::RotatedRect();
}

cv::RotatedRect MainWindow::Rotate(cv::Mat &src)
{

}

bool MainWindow::VerifySizes(cv::RotatedRect mr)
{
     //China car plate size: 440mm*140mm，aspect 3.142857

    float error = 0.3;
    float aspect = 3.142857;

     //Set a min and max area. All other patchs are discarded
    int min = 1 * aspect * 1;        // minimum area
    int max = 2000 * aspect * 2000; // maximum area

    //Get only patchs that match to a respect ratio.

    float rmin = aspect - aspect*error;
    float rmax = aspect + aspect*error;

    int area = mr.size.height * mr.size.width;
    float r = (float)mr.size.width / (float)mr.size.height;
    if (r < 1)
    {
        r = (float)mr.size.height / (float)mr.size.width;
    }
    if ((area < min || area > max) || (r < rmin || r > rmax))
    {
        return false;
    }
    else
    {
        return true;
    }
}


// 对图形求轮廓，因为车牌是一个矩形，所以可以求取轮廓，算某个矩形的倾斜角度。
void MainWindow::ImageCorrection_Contour()
{

    cv::Mat dst;

    // 一般照片都有噪点，所以可以采用高斯滤波对噪声进行消除
    dst = Gaussian(m_RawImage);
    //cv::imshow("GaussianBlur", dst);

    // 变成灰度图
    dst = Grayscale(dst);
    //cv::imshow("Gray", dst);

    // 采用 OTSU + THRESH_BINARY 算法进行二值化
    dst = TwoValued(dst);
    //cv::imshow("Threshold", dst);

    // Sobel算子（X方向）：
    // 车牌定位的核心算法，水平方向上的边缘检测，检测出车牌区域。
    dst = Sobel(dst);
    //cv::imshow("Sobel", dst);

////    // 形态学处理
////    // 闭运算：闭合起来 采用矩形获取内核
    dst = Close(dst);
    //cv::imshow("Close", dst);

    // 轮廓检测
    cv::RotatedRect rect = Contour(dst);
    cv::Rect box = rect.boundingRect();

    //cv::rectangle(m_MidImage, box, cv::Scalar(0, 0, 255));
    //cv::imshow("Contour", m_MidImage);

    dst = m_MidImage(box);

    //旋转
    cv::Point2f center( (float)(dst.cols/2) , (float) (dst.rows/2));
    cv::Mat affine_matrix = getRotationMatrix2D( center, rect.angle, 1.0 );//求得旋转矩阵
    warpAffine(dst, dst, affine_matrix, dst.size());

    // dst  => 字符
    //dst = TwoValued(dst);
    cv::Mat src = dst;
    cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);
    cv::threshold(dst, dst, 150, 255, cv::THRESH_BINARY);

    cv::imshow("threshold", dst);

    // 腐蚀 膨胀 开运算，
    cv::Mat kenal = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(dst, dst, cv::MORPH_ERODE , kenal);
    kenal = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(dst, dst, cv::MORPH_DILATE , kenal);

    cv::imshow("MORPH_ERODE", dst);

    std::vector<std::vector<cv::Point> > vContours;
    std::vector<cv::Vec4i> hierarchy;

    /*
     *  RETR_EXTERNAL:表示只检测最外层轮廓，对所有轮廓设置hierarchy[i][2]=hierarchy[i][3]=-1
        RETR_LIST:提取所有轮廓，并放置在list中，检测的轮廓不建立等级关系
        RETR_CCOMP:提取所有轮廓，并将轮廓组织成双层结构(two-level hierarchy),顶层为连通域的外围边界，次层位内层边界
        RETR_TREE:提取所有轮廓并重新建立网状轮廓结构
        RETR_FLOODFILL：官网没有介绍，应该是洪水填充法

        CHAIN_APPROX_NONE：获取每个轮廓的每个像素，相邻的两个点的像素位置差不超过1
        CHAIN_APPROX_SIMPLE：压缩水平方向，垂直方向，对角线方向的元素，值保留该方向的重点坐标，如果一个矩形轮廓只需4个点来保存轮廓信息
        CHAIN_APPROX_TC89_L1和CHAIN_APPROX_TC89_KCOS使用Teh-Chinl链逼近算法中的一种
    */

//    cv::findContours(dst, vContours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//    cv::findContours(dst, vContours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    cv::findContours(dst, vContours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
//      cv::findContours(dst, vContours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
//    cv::findContours(dst, vContours, hierarchy, cv::RETR_FLOODFILL, cv::CHAIN_APPROX_NONE);

    double dPlateArea = dst.cols * dst.rows;

    for (size_t i = 0; i < vContours.size(); i++)
    {
        double dContourArea = cv::contourArea(vContours[i]);

        double dScale = dContourArea / dPlateArea;
        if( dScale > 0.02 && dScale < 0.3 )
        {
            cv::RotatedRect rect = cv::minAreaRect(vContours[i]);

            //drawContours(src, vContours, (int)i, color, 1, cv::LINE_AA, hierarchy, 0);
            //cv::rectangle(src, rect.boundingRect(), cv::Scalar(0, 255, 0));
            //cv::putText(src, QString("%1").arg(i).toLocal8Bit().data(), rect.center, 0, 0.5,  cv::Scalar(0, 255, 0));

            cv::Mat matChar = src(rect.boundingRect());
            //cv::imshow(QString("字符:%1").arg(i).toLocal8Bit().data(), matChar);

        }
    }

    cv::imshow("src", src);

    LabelDisplayMat(ui->labelMidImage, dst);
}

// 校正的结果就是横平竖直，所以，可以采用霍夫线的方式对图像进行检测，然后所有直线求一个平均倾斜角度，在旋转角度做运算。
void MainWindow::ImageCorrection_HoughLine()
{

}

