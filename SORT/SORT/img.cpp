#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <cmath>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/data_transformer.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

#ifdef USE_OPENCV
cv::Mat RandomAffineTransform(const cv::Mat &cv_img_origin);
cv::Mat RandomScaleJittering(const cv::Mat &cv_img_origin, const int width, const int height);
cv::Mat RandomContrastJittering(const cv::Mat &cv_img_origin, const float ratio);
cv::Mat RandomContrastJittering(const cv::Mat &cv_img_origin);
cv::Mat RandomColorJittering(const cv::Mat &cv_img_origin, const float magnitude);
cv::Mat RandomColorJittering(const cv::Mat &cv_img_origin);
cv::Mat RandomPCAJittering(const cv::Mat &cv_img_origin);
cv::Mat RandomScaleCenterCropResizing(const cv::Mat &cv_img_origin, const int crop_width, const int crop_height);
cv::Mat RandomShrinkCenterCropResizing(const cv::Mat &cv_img_origin, const int width, const int height);
cv::Mat RandomInceptionAug(const cv::Mat &cv_img_origin, const int width, const int height);
cv::Mat RandomSuperAug(const cv::Mat &cv_img_origin, const int width, const int height, const int max_id, const int ratio);
#endif  // USE_OPENCV

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color,
    const int interpolation, const int resize_mode, 
    const bool super_aug, const int max_id, const int ratio, const bool use_pca) {
  cv::Mat cv_img, cv_img_warp;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  float short_side_length, scale;
  int nh, nw, mode, interp;
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  
  float interp_ratio;
  caffe_rng_uniform(1, 0.01f, 3.99f, &interp_ratio);
  // if -1,  use random interpolation for images
  interp = (interpolation == -1) ? static_cast<int> (interp_ratio) : interpolation;
  
  mode = resize_mode;
  DLOG(INFO) << "resize mode: " << mode << ", interpolation mode: " << interp;
  // compatible fix for old prototxt
  if ((height*width) == 0 && (height + width) > 0) {
      mode = 1;
  }
  if (super_aug) mode = 9;
  switch (mode)
  {
  case 0:
      if (height > 0 && width > 0) {
          cv::resize(cv_img_origin, cv_img, cv::Size(width, height), 0, 0, interp);
      } 
      else 
      {
          cv_img = cv_img_origin;
      }
      break;
  case 1:
      if (height > 0 && width > 0) {
          short_side_length = std::min(height, width);
      }
      else
      {
          short_side_length = (float)(height + width);
      }
      CHECK_GT(short_side_length, 0) << "The short side length of images must be greater than 0";
      scale = std::min(cv_img_origin.rows, cv_img_origin.cols) / short_side_length;
      nh = (int)(cv_img_origin.rows / scale);
      nw = (int)(cv_img_origin.cols / scale);
      cv::resize(cv_img_origin, cv_img, cv::Size(nw, nh), 0, 0, interp);
      break;
  case 2:
      CHECK_GT(height, 0) << "The height of images must be greater than 0";
      CHECK_GT(width, 0) << "The width of images must be greater than 0";
      cv_img = RandomScaleJittering(cv_img_origin, width, height);
      break;
  case 3:
      CHECK_GT(height, 0) << "The height of images must be greater than 0";
      CHECK_GT(width, 0) << "The width of images must be greater than 0";
      cv_img = RandomInceptionAug(cv_img_origin, width, height);
      break; 
  case 4:
      CHECK_GT(height, 0) << "The height of images must be greater than 0";
      CHECK_GT(width, 0) << "The width of images must be greater than 0";
      cv_img_warp = RandomScaleJittering(cv_img_origin, width, height);
      cv_img = RandomColorJittering(cv_img_warp);
      break;
  case 5:
      CHECK_GT(height, 0) << "The height of images must be greater than 0";
      CHECK_GT(width, 0) << "The width of images must be greater than 0";
      cv_img_warp = RandomScaleJittering(cv_img_origin, width, height);
      cv_img = RandomAffineTransform(cv_img_warp); 
      break;
  case 9:
      CHECK_GT(height, 0) << "The height of images must be greater than 0";
      CHECK_GT(width, 0) << "The width of images must be greater than 0";
      cv_img = RandomSuperAug(cv_img_origin, width, height, max_id, ratio);
      break;
  default:
      LOG(FATAL) << "Unknown resizing mode.";
  }
  if (use_pca) cv_img = RandomPCAJittering(cv_img);
  return cv_img;
}
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color,
    const int interpolation, const int resize_mode, 
    const bool super_aug, const int max_id, const int ratio) {
    return ReadImageToCVMat(filename, height, width, is_color, interpolation, resize_mode, super_aug, max_id, 5, false);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color,
    const int interpolation, const int resize_mode, 
    const bool super_aug, const int max_id) {
    return ReadImageToCVMat(filename, height, width, is_color, interpolation, resize_mode, super_aug, max_id, 5);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color,
    const int interpolation, const int resize_mode, 
    const bool super_aug) {
    return ReadImageToCVMat(filename, height, width, is_color, interpolation, resize_mode, false, 0, 0);
}
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color,
    const int interpolation, const int resize_mode) {
    return ReadImageToCVMat(filename, height, width, is_color, interpolation, resize_mode, false, 0, 0);
}
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color,
    const int interpolation) {
    return ReadImageToCVMat(filename, height, width, is_color, interpolation, 0);
}
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
    return ReadImageToCVMat(filename, height, width, is_color, 1);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}
#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

cv::Mat RandomAffineTransform(const cv::Mat &cv_img_origin)
{
    float angle, aspect_ratio, shear, translation, a, b;
    float hs, ws, ht, wt, hr, wr, scale;
    cv::Mat cv_img, transform;

    const int img_height = cv_img_origin.rows;
    const int img_width = cv_img_origin.cols;
    caffe_rng_uniform(1, -10.0f, 10.0f, &angle);
    caffe_rng_uniform(1, 0.8f, 1.25f, &scale);
    caffe_rng_uniform(1, 0.75f, 1.33f, &aspect_ratio);
    a = cos(angle / 180.0 * CV_PI);
    b = sin(angle / 180.0 * CV_PI);
    hs = 2 * scale / (1 + aspect_ratio);
    ws = aspect_ratio * hs;

    caffe_rng_uniform(1, -0.05f, 0.05f, &translation);
    ht = img_height * translation;
    caffe_rng_uniform(1, -0.05f, 0.05f, &translation);
    wt = img_width * translation;
    caffe_rng_uniform(1, -0.05f, 0.05f, &shear);
    hr = shear;
    caffe_rng_uniform(1, -0.05f, 0.05f, &shear);
    wr = shear;
    cv::Mat T_shear = (cv::Mat_<float>(2, 2) << 1, wr, hr, 1);
    cv::Mat T_scale = (cv::Mat_<float>(2, 2) << hs, 0, 0, ws);
    cv::Mat T_theta = (cv::Mat_<float>(2, 2) << a, b, -b, a);
    cv::Mat T_all = T_theta * T_scale * T_shear;

    // add random translation
    cv::Mat raw_center = (cv::Mat_<float>(2, 1) << img_width / 2.0, img_height / 2.0);
    raw_center = raw_center + (cv::Mat_<float>(2, 1) << wt, ht);
    cv::Mat new_center = (cv::Mat_<float>(2, 1) << img_width / 2.0, img_height / 2.0);
    cv::Mat T_trans = new_center - T_all * raw_center;

    vector<cv::Mat> matrix_list;
    matrix_list.push_back(T_all);
    matrix_list.push_back(T_trans);
    hconcat(matrix_list, transform);
    cv::warpAffine(cv_img_origin, cv_img, transform, cv_img_origin.size());

    return cv_img;
}

cv::Mat RandomScaleJittering(const cv::Mat &cv_img_origin, const int width, const int height)
{
    cv::Mat cv_img;
    float rand_length, origin_ratio;
    int min_short_length, max_short_length, nh, nw, interp;
    min_short_length = std::min(height, width);
    max_short_length = std::max(height, width);
    // rand_length = min_short_length + Rand(max_short_length - min_short_length + 1);
    caffe_rng_uniform(1, (float)min_short_length, (float)max_short_length, &rand_length);
    origin_ratio = static_cast<float> (cv_img_origin.cols) / static_cast<float> (cv_img_origin.rows);
    if (origin_ratio < 1) // width is short
    {
        nw = static_cast<int> (rand_length);
        nh = static_cast<int> (nw / origin_ratio);
    }
    else // height is short
    {
        nh = static_cast<int> (rand_length);
        nw = static_cast<int> (nh * origin_ratio);
    }
    if ((nh != cv_img_origin.rows) && (nw != cv_img_origin.cols))
    {
        interp = GetInterpolationMethod(cv_img_origin.cols, cv_img_origin.rows, nw, nh);    
        cv::resize(cv_img_origin, cv_img, cv::Size(nw, nh), 0, 0, interp);
        DLOG(INFO) << "The image is resized to the height and width of " << nh << "x" << nw << " with interp " << interp;
        return cv_img;
    }
    else
    {
        return cv_img_origin;
    }
}

cv::Mat RandomContrastJittering(const cv::Mat &cv_img_origin, const float ratio)
{
    cv::Mat cv_img;
    float prob, alpha;
    caffe_rng_uniform(1, 0.0f, 1.0f, &prob);
    if (prob > 0.5) {
        caffe_rng_uniform(1, 1-ratio, 1+ratio, &alpha);
        cv_img_origin.convertTo(cv_img, -1, alpha, 0.0);
    }
    else
    {
        cv_img = cv_img_origin;
    }
    return cv_img;
}

cv::Mat RandomContrastJittering(const cv::Mat &cv_img_origin)
{
    return RandomContrastJittering(cv_img_origin, 0.25f);
}

cv::Mat RandomColorJittering(const cv::Mat &cv_img_origin, const float magnitude)
{
    cv::Mat cv_img;
    vector<cv::Mat> channels;
    float prob, offset;
    cv::split(cv_img_origin, channels);

    caffe_rng_uniform(1, 0.0f, 1.0f, &prob);
    if (prob > 0.5)
    {
        for (int i = 0; i < cv_img.channels(); ++i)
        {
            caffe_rng_uniform(1, -magnitude, magnitude, &offset);
            caffe_rng_uniform(1, 0.0f, 1.0f, &prob);
            if (prob > 0.5)
            {
                channels.at(i) += offset;
            }
        }
        cv::merge(channels, cv_img);
    }
    else
    {
        cv_img = cv_img_origin;
    }

    return cv_img;
}

cv::Mat RandomColorJittering(const cv::Mat &cv_img_origin)
{
    return RandomColorJittering(cv_img_origin, 15.f);
}

cv::Mat RandomPCAJittering(const cv::Mat &cv_img_origin)
{
    float a[3], offset[3];
    
    caffe_rng_gaussian(3, 0.0f, 0.1f, a);
    a[0] *= 55.46; a[1] *= 4.79; a[2] *= 1.15;
    offset[0] = -0.5836 * a[0] - 0.6948 * a[1] + 0.4203 * a[2];
    offset[1] = -0.5808 * a[0] - 0.0045 * a[1] - 0.8140 * a[2];
    offset[2] = -0.5675 * a[0] + 0.7192 * a[1] + 0.4009 * a[2];
    DLOG(INFO) << "PCA Jittering with BGR offset: " <<
        offset[0] << ", " << offset[1] << ", " << offset[2];
    cv_img_origin += cv::Scalar(offset[0], offset[1], offset[2]);
 
    return cv_img_origin;
}
cv::Mat RandomScaleCenterCropResizing(const cv::Mat &cv_img_origin, const int crop_width, const int crop_height)
{
    cv::Mat cv_img, cv_img_cropped, cv_img_resized;
    float scale, short_side_length, ratio;
    int nh, nw, offw, offh, interp, crop_size;
    crop_size = std::min(crop_width, crop_height);
    caffe_rng_uniform(1, 1.1f, 1.5f, &ratio);
    short_side_length = crop_size * ratio; // randomly extend to 110%-150% length of crop size
    scale = std::min(cv_img_origin.rows, cv_img_origin.cols) / short_side_length;
    nh = static_cast<int>(cv_img_origin.rows / scale);
    nw = static_cast<int>(cv_img_origin.cols / scale);
    interp = (scale > 1) ? 3 : 2; // 2=bicubic for enlarge, 3=area for shrink
    cv::resize(cv_img_origin, cv_img, cv::Size(nw, nh), 0, 0, interp); 
    offw = (nw - crop_size) / 2;
    offh = (nh - crop_size) / 2;
    cv::Rect roi(offw, offh, crop_size, crop_size);
    cv_img_cropped = cv_img(roi);
    interp = GetInterpolationMethod(nw, nh, crop_width, crop_height);
    cv::resize(cv_img_cropped, cv_img_resized, cv::Size(crop_width, crop_height), 0, 0, interp);
    return cv_img_resized;
}

cv::Mat RandomShrinkCenterCropResizing(const cv::Mat &cv_img_origin, const int width, const int height)
{
    cv::Mat cv_img;
    float origin_ratio, target_ratio;
    int nh, nw, interp, h_off, w_off;
    caffe_rng_uniform(1, 0.8f, 1.0f, &target_ratio);
    origin_ratio = static_cast<float> (cv_img_origin.cols) / static_cast<float> (cv_img_origin.rows);
    if (origin_ratio < 1) // width is short
    {
        nw = static_cast<int> (cv_img_origin.cols * target_ratio);
        caffe_rng_uniform(1, origin_ratio, 1.0f, &target_ratio);
        nh = static_cast<int> ((nw - 1) / target_ratio);
    }
    else // height is short
    {
        nh = static_cast<int> (cv_img_origin.rows * target_ratio);
        caffe_rng_uniform(1, 1.0f, origin_ratio, &target_ratio);
        nw = static_cast<int> ((nh - 1) * target_ratio);
    }
    CHECK_LE(nw, cv_img_origin.cols);
    CHECK_LE(nh, cv_img_origin.rows);
    h_off = (cv_img_origin.rows - nh) / 2;
    w_off = (cv_img_origin.cols - nw) / 2;
    CHECK_LE(w_off + nw, cv_img_origin.cols) << "Cropped width should be <= image width";
    CHECK_LE(h_off + nh, cv_img_origin.rows) << "Cropped height should be <= image height";
    DLOG(INFO) << "The image is resized to the height and width of " << nh << "x" << nw;
    cv::Rect roi(w_off, h_off, nw, nh);
    cv::Mat cv_cropped_img = cv_img_origin(roi);
    interp = GetInterpolationMethod(cv_cropped_img.cols, cv_cropped_img.rows, width, height);
    cv::resize(cv_cropped_img, cv_img, cv::Size(width, height), 0, 0, interp);
    return cv_img;
}

cv::Mat RandomInceptionAug(const cv::Mat &cv_img_origin, const int width, const int height)
{
    cv::Mat cv_img;
    float area_ratio, target_ratio, target_area, prob;
    int nh, nw, interp, h_off, w_off, attempt;
    for (attempt = 0; attempt < 10; ++attempt)
    {
        caffe_rng_uniform(1, 0.15f, 1.0f, &area_ratio);
        caffe_rng_uniform(1, 0.75f, 1.3333f, &target_ratio); // [3/4, 4/3]
        target_area = cv_img_origin.rows * cv_img_origin.cols * area_ratio; //[0.15, 1.0] * area
        nw = static_cast<int> (std::sqrt(target_area * target_ratio) + 0.5f);
        nh = static_cast<int> (std::sqrt(target_area / target_ratio) + 0.5f);
        caffe_rng_uniform(1, 0.0f, 1.0f, &prob);
        if (prob > 0.5)
        {
            int tmp = nw;
            nw = nh;
            nh = tmp;
        }
        if ((nh > 1) && (nw > 1) && (nh <= cv_img_origin.rows) && (nw <= cv_img_origin.cols))
        {
            float tmp_off;
            caffe_rng_uniform(1, 0.0f, float(cv_img_origin.rows - nh + 0.99f), &tmp_off);
            h_off = static_cast<int> (tmp_off);
            caffe_rng_uniform(1, 0.0f, float(cv_img_origin.cols - nw + 0.99f), &tmp_off);
            w_off = static_cast<int> (tmp_off);
            cv::Rect roi(w_off, h_off, nw, nh);
            cv::Mat cv_cropped_img = cv_img_origin(roi);
            if ((cv_cropped_img.cols == nw) && (cv_cropped_img.rows == nh)) {
                DLOG(INFO) << "The image is of size " << cv_img_origin.rows << "x" << cv_img_origin.cols
                    << ", cropped to " << nh << "x" << nw << " with offset " << h_off << "x" << w_off;
                interp = GetInterpolationMethod(cv_cropped_img.cols, cv_cropped_img.rows, width, height);
                cv::resize(cv_cropped_img, cv_img, cv::Size(width, height), 0, 0, interp);
                return cv_img;
            }
        }
    }
    return RandomScaleCenterCropResizing(cv_img_origin, width, height);
}

cv::Mat RandomSuperAug(const cv::Mat &cv_img_origin, const int width, const int height, const int max_id, const int ratio)
{
    cv::Mat cv_img;
    float area_ratio, origin_ratio, target_ratio, target_area, prob, ph, pw, range;
    int nh, nw, interp, h_off, w_off, attempt;
    range = 1.0f / (float)ratio;
    origin_ratio = static_cast<float> (cv_img_origin.cols) / static_cast<float> (cv_img_origin.rows);
    for (attempt = 0; attempt < 10; ++attempt)
    {
        caffe_rng_uniform(1, 0.2f, 1.0f, &area_ratio);
        caffe_rng_uniform(1, 0.75f, 1.3333f, &target_ratio); // [3/4, 4/3]
        target_area = cv_img_origin.rows * cv_img_origin.cols * area_ratio; //[0.15, 1.0] * area
        caffe_rng_uniform(1, 0.0f, 1.0f, &prob);
        if (prob > 0.5)
        {
            target_ratio = 1.0f / target_ratio;
        }
        if ((origin_ratio < 0.625f) || (origin_ratio > 1.6f)) {
            if (((origin_ratio < 1) && (target_ratio > 1)) || ((origin_ratio > 1) && (target_ratio < 1))) {
                target_ratio = 1.0f / target_ratio;
            }
        }
        nw = static_cast<int> (std::sqrt(target_area * target_ratio) + 0.5f);
        nh = static_cast<int> (std::sqrt(target_area / target_ratio) + 0.5f);
        if ((nh > 0) && (nw > 0) && (nh <= cv_img_origin.rows) && (nw <= cv_img_origin.cols))
        {
            ph = (max_id / 8) * 0.125f;
            pw = (max_id % 8) * 0.125f;
            caffe_rng_uniform(1, 0.0f, 1.0f, &prob);
            ph = (ph - range) + 2 * range * prob;
            caffe_rng_uniform(1, 0.0f, 1.0f, &prob);
            pw = (pw - range) + 2 * range * prob;
            ph = (ph < 0) ? 0 : ph;
            pw = (pw < 0) ? 0 : pw;
            ph = (ph > 1) ? 1 : ph;
            pw = (pw > 1) ? 1 : pw;
            DLOG(INFO) << "area: " << area_ratio << ", aspect: " << target_ratio
                << " ph & pw: " << ph << "/" << pw << " nh & nw: " << nh << "/" << nw;
            h_off = std::max(1, static_cast<int> (cv_img_origin.rows * ph - nh / 2));
            w_off = std::max(1, static_cast<int> (cv_img_origin.cols * pw - nw / 2));
            CHECK((w_off >= 0) && (w_off < cv_img_origin.cols)) << "width offset should be >= 0 && < image width.";
            CHECK((h_off >= 0) && (h_off < cv_img_origin.rows)) << "height offset should be >= 0 && < image height.";
            nh = std::min(nh, cv_img_origin.rows - h_off);
            nw = std::min(nw, cv_img_origin.cols - w_off);
            CHECK_LE(w_off + nw, cv_img_origin.cols) << "Cropped width should be <= image width";
            CHECK_LE(h_off + nh, cv_img_origin.rows) << "Cropped height should be <= image height";
            DLOG(INFO) << "h_off & w_off: " << h_off << "/" << w_off << ", nh & nw: " << nh << "/" << nw;
            cv::Rect roi(w_off, h_off, nw, nh);
            cv::Mat cv_cropped_img = cv_img_origin(roi);
            if ((cv_cropped_img.cols == nw) && (cv_cropped_img.rows == nh)) {
                DLOG(INFO) << "The image is of size " << cv_img_origin.rows << "x" << cv_img_origin.cols
                    << ", cropped to " << nh << "x" << nw << " with offset " << h_off << "x" << w_off
                    << ", attempt: " << attempt;
                interp = GetInterpolationMethod(cv_cropped_img.cols, cv_cropped_img.rows, width, height);
                cv::resize(cv_cropped_img, cv_img, cv::Size(width, height), 0, 0, interp);
                return cv_img;
            }
        }
    }
    return RandomShrinkCenterCropResizing(cv_img_origin, width, height);
}
#endif  // USE_OPENCV

}  // namespace caffe
