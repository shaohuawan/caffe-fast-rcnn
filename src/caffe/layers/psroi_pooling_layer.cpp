// ------------------------------------------------------------------
// R-FCN
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/psroi_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    PSROIPoolingParameter psroi_pooling_param =
      this->layer_param_.psroi_pooling_param();
    spatial_scale_ = psroi_pooling_param.spatial_scale();
    LOG(INFO) << "Spatial scale: " << spatial_scale_;

    CHECK_GT(psroi_pooling_param.output_dim(), 0)
      << "output_dim must be > 0";
    CHECK_GT(psroi_pooling_param.group_size(), 0)
      << "group_size must be > 0";

    output_dim_ = psroi_pooling_param.output_dim();
    group_size_ = psroi_pooling_param.group_size();
    pooled_height_ = group_size_;
    pooled_width_ = group_size_;
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    channels_ = bottom[0]->channels();
    CHECK_EQ(channels_, output_dim_*group_size_*group_size_)
      << "input channel number does not match layer parameters";
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(
      bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
    mapping_channel_.Reshape(
      bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_rois = bottom[1]->cpu_data();
    // Number of ROIs
    int num_rois = bottom[1]->num();
    int batch_size = bottom[0]->num();
    int top_count = top[0]->count();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_cpu_data();
    caffe_set(top_count, Dtype(0), top_data);
    caffe_set(top_count, -1, mapping_channel_ptr);

    for(int n = 0; n < num_rois; ++n) {
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w =
        static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale_;
      Dtype roi_start_h =
        static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale_;
      Dtype roi_end_w =
        static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale_;
      Dtype roi_end_h =
        static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale_;
      CHECK_GE(roi_batch_ind, 0);
      CHECK_LT(roi_batch_ind, batch_size);

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, static_cast<Dtype>(0.1));  // avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, static_cast<Dtype>(0.1));

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height_);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width_);

      const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);
      
      std::vector<int> vhstart, vwstart, vhend, vwend;

      for(int ph = 0; ph < pooled_height_; ++ph) {
        for(int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = floor(static_cast<Dtype>(ph) * bin_size_h
                          + roi_start_h);
            int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
                          + roi_start_w);
            int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
                        + roi_start_h);
            int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
                        + roi_start_w);
            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart, 0), height_);
            hend = min(max(hend, 0), height_);
            wstart = min(max(wstart, 0), width_);
            wend = min(max(wend, 0), width_);

            vhstart.push_back(hstart);
            vwstart.push_back(wstart);
            vhend.push_back(hend);
            vwend.push_back(wend);
        }
      }

      for(int c = 0; c < output_dim_; ++c) {
        for(int ph = 0; ph < pooled_height_; ++ph) {
          for(int pw = 0; pw < pooled_width_; ++pw) {
            const int pool_index = ph * pooled_width_ + pw;
            int hstart = vhstart[pool_index];
            int hend = vhend[pool_index];
            int wstart = vwstart[pool_index];
            int wend = vwend[pool_index];
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            Dtype out_sum = 0;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                int index = h*width_ + w;
                out_sum += batch_data[index];
              }
            }

            Dtype bin_area = (hend - hstart)*(wend - wstart);
            top_data[pool_index] = is_empty? 0. : out_sum/bin_area;
            //mapping_channel[pool_index] = c;
            
            batch_data += bottom[0]->offset(0, 1);
          }
        }
        top_data += top[0]->offset(0, 1);
        //mapping_channel += top[0]->offset(0, 1);
      }

      // Increment ROI data pointer
      bottom_rois += bottom[1]->offset(1);
    }

  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
#ifdef CPU_ONLY
  STUB_GPU(PSROIPoolingLayer);
#endif

  INSTANTIATE_CLASS(PSROIPoolingLayer);
  REGISTER_LAYER_CLASS(PSROIPooling);

}  // namespace caffe
