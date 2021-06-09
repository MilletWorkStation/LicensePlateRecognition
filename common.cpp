#include "common.h"

//参数1-显示图像的Label，参数2-要显示的Mat
void LabelDisplayMat(QLabel *label, cv::Mat &mat)
{
    cv::Mat Rgb;
    QImage Img;
    if (mat.channels() == 3)//RGB Img
    {
        cv::cvtColor(mat, Rgb, CV_BGR2RGB);//颜色空间转换
        Img = QImage((const uchar*)(Rgb.data), Rgb.cols, Rgb.rows, Rgb.cols * Rgb.channels(), QImage::Format_RGB888);
    }
    else//Gray Img
    {
        Img = QImage((const uchar*)(mat.data), mat.cols, mat.rows, mat.cols*mat.channels(), QImage::Format_Indexed8);
    }

    QPixmap pix = QPixmap::fromImage(Img);

    //pix = pix.scaled(label->width(), label->height());

    label->setPixmap(pix);
}

cv::Mat LoadImage(QString str)
{
    cv::Mat mat = cv::imread(str.toLocal8Bit().data());

    if( mat.data == NULL )
    {
        QMessageBox::warning(nullptr, "错误", "载入图片错误");
    }

    return mat;
}
