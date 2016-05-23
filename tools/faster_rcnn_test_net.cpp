#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/misc.hpp"
#include "opencv/cv.h"
#include "opencv2/highgui.hpp"
#include <boost/timer/timer.hpp>

#include <iostream>
#include <fstream>

using namespace boost::filesystem;

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;

using caffe::Phase;
using caffe::Box;
using caffe::keep;
using caffe::nms;

using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(bottom_vec, &iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(vector<Blob<float>*>(), &initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);


//-----------  Shaohua Wan - Begin  -------------------

#define DEBUG

static void CheckFile(const string& filename) {
    std::ifstream f(filename.c_str());
    if (!f.good()) {
      f.close();
      throw std::runtime_error("Could not open file " + filename);
    }
    f.close();
}

template<typename Dtype>
static shared_ptr<Net<Dtype> > Net_Init_Load(
    string param_file, string pretrained_param_file, int phase) {
  CheckFile(param_file);
  CheckFile(pretrained_param_file);

  shared_ptr<Net<Dtype> > net(new Net<Dtype>(param_file, static_cast<caffe::Phase>(phase)));
  net->CopyTrainedLayersFrom(pretrained_param_file);
  return net;
}

template<typename Dtype>
void test_net( shared_ptr<Net<Dtype> > net, const std::vector<std::string>& imdb )
{
    std::string imdb_path = "";
    for(int i=0; i<imdb.size(); i++)
    {
        cv::Mat im = cv::imread( imdb_path + imdb.at(i) );
    }
}

template<typename Dtype>
void vis_detections_print_to_file(cv::Mat& im, const string& image_name, 
    const string& class_name, const std::vector<Box<Dtype> >& boxes, 
    const std::vector<Dtype>& scores, float thresh = 0.5)
{
    //Draw detected bounding boxes.
    std::vector<int> inds;
    for(int i=0; i<scores.size(); i++)
    {
        if(scores[i]>=thresh)
            inds.push_back(i);
    }

    if( inds.size() == 0 ) return;

    for(int i=0; i<inds.size(); i++)
    {
        Box<Dtype> bbox = boxes[ inds[i] ];
        Dtype score = scores[ inds[i] ]; 

        int x_ = (int) bbox.x1;
        int y_ = (int) bbox.y1; 
        int w_ = (int) bbox.x2 - x_ + 1;
        int h_ = (int) bbox.y2 - y_ + 1;
        cv::Rect rect(x_, y_, w_, h_);
        cv::Scalar color(0,255,0);
        cv::rectangle(im, rect, color, 3);
        ostringstream text; 
        text << class_name << score;
        cv::putText( im, text.str(), cv::Point( x_, y_ ), 
                     CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0) );
    }

    string im_name = image_name+"."+class_name+".png";
//    cv::imwrite(im_name, im);
}

template<typename Dtype>
void vis_detect(cv::Mat& im, shared_ptr<Blob<Dtype> > scores, 
                const std::vector<Box<Dtype> >& pred_boxes)
{
    Dtype maxScore = 0;
    int maxI = 0;
    int maxJ = 0;

    int n = scores->num();
//    int c = scores->channels();
//    int h = scores->height();
//    int w = scores->width();

    for(int i=0; i<n; i++)
    {
        for(int j=1; j<IMDB_NUM_CLS; j++)  // IMDB_NUM_CLS: background class plus foreground object classes
        {
            if(scores->cpu_data()[ scores->offset(i,j,0,0) ] > maxScore)
            {
                maxScore = scores->cpu_data()[ scores->offset(i,j,0,0) ];
                maxI     = i;
                maxJ     = j;
            }
        }
    }

    int k = maxI*IMDB_NUM_CLS + maxJ;
    int x_ = (int) pred_boxes[k].x1;
    int y_ = (int) pred_boxes[k].y1; 
    int w_ = (int) pred_boxes[k].x2 - x_ + 1;
    int h_ = (int) pred_boxes[k].y2 - y_ + 1;
    cv::Rect rect(x_, y_, w_, h_);
    cv::Scalar color(0,255,0);
    cv::rectangle(im, rect, color, 1);
    
    cv::imwrite("result.png", im);
}

template<typename Dtype>
void demo_print_to_file(shared_ptr<Net<Dtype> > net, 
                                       const string& image_name)
{
    std::vector<string> CLASSES;
    CLASSES.push_back("__background__");
    CLASSES.push_back("face");

    // Load the demo image
    const string im_file = image_name;
    cv::Mat im = cv::imread(im_file);
    std::vector<Box<float> > boxes, pred_boxes;
    shared_ptr<Blob<float> > scores; // scores and pred_boxes for foreground objects and background

    // Detect all object classes and regress object bounds
    {
        boost::timer::auto_cpu_timer t;
        scores = caffe::im_detect<Dtype>(net, im, boxes, pred_boxes);
    }

    // The following prints all detections 
    float CONF_THRESH = 0.3;
    float NMS_THRESH = 0.3;

    int n = scores->num();
//    int c = scores->channels();
//    int h = scores->height();
//    int w = scores->width();

    for(int j=1; j<IMDB_NUM_CLS; j++)  // IMDB_NUM_CLS: background class plus foreground object classes
    {
        std::vector<Box<Dtype> > cls_boxes;
        std::vector<Dtype> cls_scores;
        for(int i=0; i<n; i++)
        {
            int k = i*IMDB_NUM_CLS + j;
            cls_boxes.push_back( pred_boxes[k] );
            Dtype score = scores->cpu_data()[ scores->offset(i,j,0,0) ];
            cls_scores.push_back( score );
        }
        std::vector<int> keep_ind = nms( cls_boxes, cls_scores, (Dtype) NMS_THRESH);
        cls_boxes  = keep( cls_boxes,  keep_ind );
        cls_scores = keep( cls_scores, keep_ind );
        cv::Mat orig_im = im;
        vis_detections_print_to_file(orig_im, image_name, 
            CLASSES[j], cls_boxes, cls_scores, CONF_THRESH);
    }

}

int main1()
// This main function was used to compare to output between
// C++-based FasterRCNN and the python-based FasterRCNN.
{
//    Caffe::set_mode(Caffe::CPU);
    Caffe::SetDevice(3);
    Caffe::set_mode(Caffe::GPU);

    string prototxt = "vgg16_faster_rcnn_test.pt";
    string caffemodel = "VGG16_faster_rcnn_final.caffemodel_hmmb_0001";
    //string prototxt = "zf_ali_faster_rcnn_test.pt";
    //string caffemodel = "ZF_faster_rcnn_final.caffemodel_ali_alllist_iter200k_basesize16_scales600";
    shared_ptr<Net<float> > net = Net_Init_Load<float>(prototxt, caffemodel, caffe::TEST);

    cv::Mat im = cv::imread("/home/wanshaohua/shaohua_package/caffe-fast-rcnn_ali/59d7958fe4c4ec2b437de95d0d9752b47.jpg");
    std::vector<Box<float> > boxes, pred_boxes;
    shared_ptr<Blob<float> > scores; // scores and pred_boxes for foreground objects and background
    scores = caffe::im_detect<float>(net, im, boxes, pred_boxes);
 
    vis_detect(im, scores, pred_boxes);

    return 0;
}

int main()
{
//    Caffe::SetDevice(3);
//    Caffe::set_mode(Caffe::GPU);
    Caffe::set_mode(Caffe::CPU);
    string prototxt = "../py-faster-rcnn_001_c++/faster_rcnn_test_svd.pt";
    string caffemodel = "../py-faster-rcnn_001_c++/VGG_CNN_M_1024_faster_rcnn_final_svd_fc6_1024_fc7_256.caffemodel";
    shared_ptr<Net<float> > net = Net_Init_Load<float>(prototxt, caffemodel, caffe::TEST);

    string im_list = "../py-faster-rcnn_001_c++/data/tmp_devkit2016/tmp2016/ImageSets/Main/test.txt";
    string im_path = "../py-faster-rcnn_001_c++/data/tmp_devkit2016/tmp2016/JPEGImages/";
    std::ifstream ifs(im_list.c_str());
    string im_name;
    while( std::getline( ifs, im_name ) )
    {
        im_name = im_path + im_name + ".jpg";
        demo_print_to_file(net, im_name);
    }

    return 0;
}
