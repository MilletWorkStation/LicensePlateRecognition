#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QAction>
#include <QFileDialog>


#include <opencv2/core/bufferpool.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>



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
    QString strFile = QFileDialog::getOpenFileName(this, QString("���ļ�").toLocal8Bit(), "", "Image Files(*.jpg *.png) *.bmp");
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

    cv::Mat xMat, yMat;
    cv::Sobel(src, xMat, -1, 1, 0);
    cv::Sobel(src, yMat, -1, 0, 1);

    cv::addWeighted(xMat, 1, yMat, 0.5, 0.5, dst);

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

        // ��߱��ڷ�Χ��
        if(VerifySizes(rect))
        {
            //drawContours(m_MidImage, vContours, (int)i, color, 1, cv::LINE_AA, hierarchy, 0);
            //cv::putText(m_MidImage, QString("%1").arg(rect.angle).toLocal8Bit().data(), vContours[i][0], 0, 0.5, cv::Scalar(0, 255, 0));
//            cv::putText(m_MidImage, QString("%1:%2").arg(i).arg(rect.angle).toLocal8Bit().data(), rect.center, 0, 0.5, cv::Scalar(0, 255, 0));

           // ������Ĵ洢
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

double MatCompare(cv::Mat &src, cv::Mat &model)               //ģ����ͼ��Աȼ������ƶ�
{
    cv::Mat re_model;
    cv::resize(model, re_model, src.size());
    int rows, cols;
    uchar *src_data, *model_data;
    rows = re_model.rows;
    cols = re_model.cols*src.channels();
    double percentage,same=0.0,different=0.0;

    for (int i = 0; i < rows; i++)       //����ͼ������
    {
        src_data = src.ptr<uchar>(i);
        model_data = re_model.ptr<uchar>(i);
        for (int j = 0; j < cols; j++)
        {
            if (src_data[j] == model_data[j])
            {
                same++;         //��¼����ֵ��ͬ�ĸ���
            }
            else
            {
                different++;    //��¼����ֵ��ͬ�ĸ���
            }
        }
    }
    percentage = same / (same + different);
    return percentage;                     //�������ƶ�
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
    cv::imshow("Threshold", dst);

    // Sobel���ӣ�X���򣩣�
    // ���ƶ�λ�ĺ����㷨��ˮƽ�����ϵı�Ե��⣬������������
    dst = Sobel(dst);
    cv::imshow("Sobel", dst);

////    // ��̬ѧ����
////    // �����㣺�պ����� ���þ��λ�ȡ�ں�
    dst = Close(dst);
    //cv::imshow("Close", dst);

    // �������
    cv::RotatedRect rect = Contour(dst);
    cv::Rect box = rect.boundingRect();

    //cv::rectangle(m_MidImage, box, cv::Scalar(0, 0, 255));
    //cv::imshow("Contour", m_MidImage);

    dst = m_MidImage(box);

    //��ת
    cv::Point2f center( (float)(dst.cols/2) , (float) (dst.rows/2));
    cv::Mat affine_matrix = getRotationMatrix2D( center, rect.angle, 1.0 );//�����ת����
    warpAffine(dst, dst, affine_matrix, dst.size());

    // dst  => �ַ�
    //dst = TwoValued(dst);
    cv::Mat src = dst;
    cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);
    cv::threshold(dst, dst, 150, 255, cv::THRESH_BINARY);

    //cv::imshow("threshold", dst);

    // ��ʴ ���� �����㣬
    cv::Mat kenal = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(dst, dst, cv::MORPH_ERODE , kenal);
    kenal = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(dst, dst, cv::MORPH_DILATE , kenal);

    //cv::imshow("MORPH_ERODE", dst);

    std::vector<std::vector<cv::Point> > vContours;
    std::vector<cv::Vec4i> hierarchy;

    /*
     *  RETR_EXTERNAL:��ʾֻ����������������������������hierarchy[i][2]=hierarchy[i][3]=-1
        RETR_LIST:��ȡ������������������list�У����������������ȼ���ϵ
        RETR_CCOMP:��ȡ��������������������֯��˫��ṹ(two-level hierarchy),����Ϊ��ͨ�����Χ�߽磬�β�λ�ڲ�߽�
        RETR_TREE:��ȡ�������������½�����״�����ṹ
        RETR_FLOODFILL������û�н��ܣ�Ӧ���Ǻ�ˮ��䷨

        CHAIN_APPROX_NONE����ȡÿ��������ÿ�����أ����ڵ������������λ�ò����1
        CHAIN_APPROX_SIMPLE��ѹ��ˮƽ���򣬴�ֱ���򣬶Խ��߷����Ԫ�أ�ֵ�����÷�����ص����꣬���һ����������ֻ��4����������������Ϣ
        CHAIN_APPROX_TC89_L1��CHAIN_APPROX_TC89_KCOSʹ��Teh-Chinl���ƽ��㷨�е�һ��
    */

//    cv::findContours(dst, vContours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//    cv::findContours(dst, vContours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    cv::findContours(dst, vContours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
//      cv::findContours(dst, vContours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
//    cv::findContours(dst, vContours, hierarchy, cv::RETR_FLOODFILL, cv::CHAIN_APPROX_NONE);

    double dPlateArea = dst.cols * dst.rows;

    std::vector<cv::Mat> vCharMat;


    cv::Mat testMat;
    for (size_t i = 0; i < vContours.size(); i++)
    {
        double dContourArea = cv::contourArea(vContours[i]);

        double dScale = dContourArea / dPlateArea;
        if( dScale > 0.02 && dScale < 0.3 )
        {
            //
            cv::RotatedRect rotRect = cv::minAreaRect(vContours[i]);

            // ��С������
            cv::Rect minRect = cv::boundingRect(vContours[i]);

            //drawContours(src, vContours, (int)i, cv::Scalar(0, 0, 255), 1, cv::LINE_AA, hierarchy, 0);

            //cv::rectangle(src, minRect, cv::Scalar(0, 255, 0));
            //cv::putText(src, QString("%1").arg(i).toLocal8Bit().data(), rect.center, 0, 0.5,  cv::Scalar(0, 255, 0));

            cv::Mat matChar = src(minRect);
            if(vCharMat.size() == 0)
                testMat = src(minRect);

            cv::resize(matChar, matChar, cv::Size(50, 30));

            cv::cvtColor(matChar, matChar, cv::COLOR_RGB2GRAY);
            cv::threshold(matChar, matChar, 100, 255, cv::THRESH_BINARY_INV);

            cv::imshow(QString("char:%1").arg(i).toLocal8Bit().data(), matChar);

            vCharMat.push_back(matChar);
        }
    }



    // ����ģ��
    kenal = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    std::vector<cv::Mat> vTemplate;
    for(int i = 0; i < 10; ++i)
    {
        cv::Mat temp = cv::imread(QString("f:\\template\\%1.png").arg(i).toLocal8Bit().data(), cv::IMREAD_GRAYSCALE);
        cv::resize(temp, temp, cv::Size(50, 30));

        cv::threshold(temp, temp, 125, 255, cv::THRESH_BINARY);
        //cv::imshow(QString("template:%1").arg(i).toLocal8Bit().data(), temp);

        cv::erode(temp, temp, kenal);

        vTemplate.push_back(temp);

    }


#if 1       // ͬ�ַ��Ƚ�


//    // ��ȡ��������
//    vContours.clear();
//    hierarchy.clear();

//    cv::resize(testMat, testMat, cv::Size(50, 30));

//    //    cv::findContours(dst, vContours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//    //    cv::findContours(dst, vContours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
//    //cv::findContours(dst, vContours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
//    //      cv::findContours(dst, vContours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
//    //    cv::findContours(dst, vContours, hierarchy, cv::RETR_FLOODFILL, cv::CHAIN_APPROX_NONE);

//    cv::findContours(vCharMat[0], vContours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
//    for (size_t i = 0; i < vContours.size(); i++)
//    {
//        cv::RotatedRect rect = cv::minAreaRect(vContours[i]);


//        drawContours(testMat, vContours, (int)i, cv::Scalar(0, 0, 255), 1, cv::LINE_AA, hierarchy, 0);
//    }

//    cv::imshow(QString("vCharMat:%1").arg(0).toLocal8Bit().data(), testMat);



    for(int i = 0; i < vCharMat.size(); ++i)
    {
        int nPerfectIndex = -1;
        double dBestMatchRate = 1.0;


//        cv::imshow(QString("vCharMat[%1]").arg(i).toLocal8Bit().data(), vCharMat[i]);
//        cv::moveWindow(QString("vCharMat[%1]").arg(i).toLocal8Bit().data(),i * 100, 0);

        for(int j = 0; j < vTemplate.size(); ++j)
        {

//            cv::Mat image_matched;
//            cv::matchTemplate(vCharMat[i], vTemplate[j], image_matched, cv::TM_CCOEFF_NORMED);
            //cv::imshow(QString("Char:%1 %2").arg(i).arg(j).toLocal8Bit().data(), image_matched);


            // ͼƬ�����
//            cv::Mat result = vCharMat[i] - vTemplate[j];
//            cv::imshow(QString("result:%1").arg(i).toLocal8Bit().data(), result);


//            double matchRate = MatCompare(vCharMat[i], vTemplate[j]);
//            if(matchRate > 0.35)
//            {
//                ui->leNumberPlate->setText(ui->leNumberPlate->text().append(QString("%1").arg(j)));
//            }


            // ��ʵ�ʾ��Ǽ�������ͼ���HU�أ�Ȼ��Ƚ�����ͼ��HU�صľ��룬����ԽС˵������ͼ��Խ���ƣ�����ֵԽ��˵������ͼ��Խ�����ơ�
            // HU�ؾ���ƽ�Ʋ����ԣ���ת�����ԣ����Ų����ԣ������ܹ���ͼ�������Ч��ƥ�䡣
            // ��ͨ��
            double dMatchRate = cv::matchShapes(vCharMat[i], vTemplate[j], 1, 0);
            if(dMatchRate < dBestMatchRate)
            {
                dBestMatchRate = dMatchRate;
                nPerfectIndex = j;
            }

            //cv::imshow(QString("vTemplate[%1]").arg(j).toLocal8Bit().data(), vTemplate[j]);
            //cv::moveWindow(QString("vTemplate[%1]").arg(j).toLocal8Bit().data(), j * 100, 300 );
        }

        if( nPerfectIndex >= 0 )
            ui->labelMidImage->setText(ui->labelMidImage->text().append( QString("%1 : %2 \n").arg(nPerfectIndex).arg(dBestMatchRate)));
    }

#endif


    //LabelDisplayMat(ui->labelMidImage, dst);
}

// У���Ľ�����Ǻ�ƽ��ֱ�����ԣ����Բ��û����ߵķ�ʽ��ͼ����м�⣬Ȼ������ֱ����һ��ƽ����б�Ƕȣ�����ת�Ƕ������㡣
void MainWindow::ImageCorrection_HoughLine()
{

}

