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
    QString strFile = QFileDialog::getOpenFileName(this, "���ļ�");
    if(strFile == "")
        return;

    m_RawImage = cv::imread(strFile.toLocal8Bit().data());
    m_MidImage = m_RawImage.clone();

    LabelDisplayMat(ui->lalbelRawImage, m_RawImage);

    Recognition();
}
/*
 * ��Ҫ����
 * 1��У��ͼƬ
 * 2���ָ�����������޸ĳߴ�
 *  ��˹�˲�
 * 3�����ƶ�ֵ������
 * 4���Զ�ֵͼ�������̬ѧ����ȡ��
 * 5��ȥ���ַ���ĸ�����Ϣ
 * 6����̬ѧ������ȡ�ַ�
 * 7���ָ��ַ�������
 * 8���ַ�ʶ��������
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

    cv::morphologyEx(src, dst, cv::MORPH_DILATE , kenal);
    //cv::morphologyEx(dst, dst, cv::MORPH_OPEN , kenal);
    return dst;
}

cv::Mat MainWindow::Contour(cv::Mat& src)
{
    std::vector<std::vector<cv::Point> > vContours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(src, vContours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    cv::Scalar color = cv::Scalar(0, 0, 255);

    for (size_t i = 0; i < vContours.size(); i++)
    {
        cv::RotatedRect rect = cv::minAreaRect(vContours[i]);

        if(VerifySizes(rect))
        {
            drawContours(m_MidImage, vContours, (int)i, color, 1, cv::LINE_AA, hierarchy, 0);
            //cv::putText(m_MidImage, QString("%1").arg(rect.angle).toLocal8Bit().data(), vContours[i][0], 0, 0.5, cv::Scalar(0, 255, 0));
            cv::putText(m_MidImage, QString("%1").arg(rect.angle).toLocal8Bit().data(), rect.center, 0, 0.5, cv::Scalar(0, 255, 0));


        }
    }

    return m_MidImage;
}

bool MainWindow::VerifySizes(cv::RotatedRect mr)
{
     //China car plate size: 440mm*140mm��aspect 3.142857

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


// ��ͼ������������Ϊ������һ�����Σ����Կ�����ȡ��������ĳ�����ε���б�Ƕȡ�
void MainWindow::ImageCorrection_Contour()
{

    cv::Mat dst;

    // һ����Ƭ������㣬���Կ��Բ��ø�˹�˲���������������
    dst = Gaussian(m_RawImage);
    //cv::imshow("GaussianBlur", dst);

    // ��ɻҶ�ͼ
    dst = Grayscale(dst);
    //cv::imshow("Gray", dst);

    // ���� OTSU + THRESH_BINARY �㷨���ж�ֵ��
    dst = TwoValued(dst);
    //cv::imshow("Threshold", dst);

    // Sobel���ӣ�X���򣩣�
    // ���ƶ�λ�ĺ����㷨��ˮƽ�����ϵı�Ե��⣬������������
    dst = Sobel(dst);
    //cv::imshow("Sobel", dst);

////    // ��̬ѧ����
////    // �����㣺�պ����� ���þ��λ�ȡ�ں�
    dst = Close(dst);
    //cv::imshow("Close", dst);

    // �������
    dst = Contour(dst);
    cv::imshow("Contour", dst);


    LabelDisplayMat(ui->labelMidImage, dst);

}

// У���Ľ�����Ǻ�ƽ��ֱ�����ԣ����Բ��û����ߵķ�ʽ��ͼ����м�⣬Ȼ������ֱ����һ��ƽ����б�Ƕȣ�����ת�Ƕ������㡣
void MainWindow::ImageCorrection_HoughLine()
{

}

