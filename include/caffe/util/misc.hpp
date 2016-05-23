#ifndef _CAFFE_UTIL_MISC_HPP_
#define _CAFFE_UTIL_MISC_HPP_

#include <cmath>
#include <caffe/blob.hpp>
#include <caffe/net.hpp>

#include "caffe/proto/caffe.pb.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/imgproc.hpp"

#define RPN_PRE_NMS_TOP_N  6000
#define RPN_POST_NMS_TOP_N 300
#define RPN_NMS_THRESH     0.7
#define RPN_MIN_SIZE       16
#define IMDB_NUM_CLS       2

namespace caffe {

template<typename Dtype>
class Box{
  public:
    Box(Dtype x1_, Dtype y1_, Dtype x2_, Dtype y2_)
      : x1(x1_), y1(y1_), x2(x2_), y2(y2_) {};
    Box(const Box& box, Dtype shift_x, Dtype shift_y)
      : x1(box.x1+shift_x), y1(box.y1+shift_y), 
        x2(box.x2+shift_x), y2(box.y2+shift_y) {};
    
    Dtype x1, y1, x2, y2;
};


template <typename Dtype>
std::vector<int> nms(const std::vector<Box<Dtype> >& boxes, 
                     const std::vector<Dtype>& scores, const Dtype& nms_thresh);

template <typename Dtype>
std::vector<Box<Dtype> > _mkanchors(const std::vector<Dtype>& ws, 
                                    const std::vector<Dtype>& hs, 
                                    Dtype x_ctr, Dtype y_ctr);

template <typename Dtype>
std::vector<Box<Dtype> > _ratio_enum(const Box<Dtype> & box,
                                     const std::vector<Dtype>& ratios);

template <typename Dtype>
std::vector<Box<Dtype> > _scale_enum(const Box<Dtype> & box,
                                     const std::vector<int>& scales);

template <typename Dtype>
std::vector<Box<Dtype> > generate_anchors(const int& base_size,
                                          const std::vector<Dtype>& ratios, 
                                          const std::vector<int>& scales );

template <typename Dtype>
bool myfunction1 (const pair<Dtype,size_t>& i, const pair<Dtype,size_t>& j) { return (i.first<j.first); }
template <typename Dtype>
bool myfunction2 (const pair<Dtype,size_t>& i, const pair<Dtype,size_t>& j) { return (i.first>j.first); }

template <typename Dtype>
std::vector<int> sort_ind( std::vector<Dtype> scores, bool increasing = true );

template <typename Dtype>
std::vector<Box<Dtype> > bbox_transform_inv(const std::vector<Box<Dtype> >& boxes, Blob<Dtype>* deltas);

template <typename Dtype>
std::vector<int> _filter_boxes( const std::vector<Box<Dtype> >& boxes, Dtype min_size );

template <typename Dtype>
void clip_boxes( std::vector<Box<Dtype> >& boxes, int im_rows, int im_cols);

template <typename T>
std::vector<T> keep( std::vector<T> items, std::vector<int> kept_ind )
{
  std::vector<T> newitems;
  for(int i=0; i<kept_ind.size(); i++)
    newitems.push_back( items[ kept_ind[i] ] );  
  return newitems;
}

template<typename Dtype>
shared_ptr<Blob<Dtype> > _get_image_blob(const cv::Mat& im, 
                                         const std::vector<float>& im_scale_factors);

template<typename Dtype>
shared_ptr<Blob<Dtype> > im_detect(  shared_ptr<Net<Dtype> > net, 
                   const cv::Mat& im, std::vector<Box<Dtype> >& boxes,
                   std::vector<Box<Dtype> >& pred_boxes);

template<typename Dtype>
shared_ptr<Blob<Dtype> > im_list_to_blob(const std::vector<cv::Mat>& ims);

template <typename Dtype>
std::vector<Box<Dtype> > bbox_transform_inv1(const std::vector<Box<Dtype> >& boxes, 
                                             shared_ptr<Blob<Dtype> > deltas);

template<typename Dtype>
void net_forward( shared_ptr<Net<Dtype> > net, const std::vector<Blob<Dtype> *>& blobs, 
                  std::string start, std::string end, 
                  shared_ptr<Blob<Dtype> > blob_data )
{
    int start_ind = 0;
    int end_ind = net->layers().size() - 1;
    net->blob_by_name("data")->CopyFrom( *blob_data );
    net->ForwardFromTo(start_ind, end_ind);
}

}  // namespace caffe

#endif  // CAFFE_UTIL_MISC_HPP_
