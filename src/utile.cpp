#include "utile.h"

/*
Mat warp_face_by_face_landmark_5(const Mat temp_vision_frame, Mat &crop_img, const vector<Point2f> face_landmark_5, const vector<Point2f> normed_template, const Size crop_size)
{
    //vector<uchar> inliers(face_landmark_5.size(), 0);
    Mat affine_matrix = cv::estimateAffinePartial2D(face_landmark_5, normed_template, cv::noArray(), cv::RANSAC, 100.0);
    warpAffine(temp_vision_frame, crop_img, affine_matrix, crop_size, cv::INTER_AREA, cv::BORDER_REPLICATE);
    return affine_matrix;
}*/
/*
Mat create_static_box_mask(const int *crop_size, const float face_mask_blur, const int *face_mask_padding)
{
    const float blur_amount = int(crop_size[0] * 0.5 * face_mask_blur);
    const int blur_area = max(int(blur_amount / 2), 1);
    Mat box_mask = Mat::ones(crop_size[0], crop_size[1], CV_32FC1);

    int sub = max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100));
    // Mat roi = box_mask(cv::Rect(0,0,sub,crop_size[1]));
    box_mask(cv::Rect(0, 0, crop_size[1], sub)).setTo(0);

    sub = crop_size[0] - max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100));
    box_mask(cv::Rect(0, sub, crop_size[1], crop_size[0] - sub)).setTo(0);

    sub = max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100));
    box_mask(cv::Rect(0, 0, sub, crop_size[0])).setTo(0);

    sub = crop_size[1] - max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100));
    box_mask(cv::Rect(sub, 0, crop_size[1] - sub, crop_size[0])).setTo(0);

    if (blur_amount > 0)
    {
        GaussianBlur(box_mask, box_mask, Size(0, 0), blur_amount * 0.25);
    }
    return box_mask;
}
*/