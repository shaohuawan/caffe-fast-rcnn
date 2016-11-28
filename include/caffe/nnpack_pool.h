#pragma once

#include <boost/noncopyable.hpp>

#include "nnpack.h"
#include "mkl.h"

namespace caffe {
class NNPACKPool : public boost::noncopyable {
 public:
   NNPACKPool() {
#ifdef USE_MKL
     const size_t num_mkl_threads = mkl_get_max_threads();
#else
     // Can we do better here?
     const size_t num_mkl_threads = 1;
#endif
     if (num_mkl_threads > 1) {
       pool_ = pthreadpool_create(num_mkl_threads);
     } else {
       pool_ = NULL;
     }

   }
  ~NNPACKPool() {
    if (pool_) {
      pthreadpool_destroy(pool_);
    }
    pool_ = NULL;
  }

  pthreadpool_t pool() { return pool_; };

 private:
  pthreadpool_t pool_;
};

}
