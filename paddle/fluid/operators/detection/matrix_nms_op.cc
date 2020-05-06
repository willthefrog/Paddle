/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detection/nms_util.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

class MatrixNMSOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("BBoxes"), "Input", "BBoxes", "MatrixNMS");
    OP_INOUT_CHECK(ctx->HasInput("Scores"), "Input", "Scores", "MatrixNMS");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "MatrixNMS");
    auto box_dims = ctx->GetInputDim("BBoxes");
    auto score_dims = ctx->GetInputDim("Scores");
    auto score_size = score_dims.size();

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(score_size == 3, true,
                        platform::errors::InvalidArgument(
                            "The rank of Input(Scores) must be 3. "
                            "But received rank = %d.",
                            score_size));
      PADDLE_ENFORCE_EQ(box_dims.size(), 3,
                        platform::errors::InvalidArgument(
                            "The rank of Input(BBoxes) must be 3."
                            "But received rank = %d.",
                            box_dims.size()));
      PADDLE_ENFORCE_EQ(
        box_dims[2] == 4,
        true, platform::errors::InvalidArgument(
          "The last dimension of Input (BBoxes) must be 4, "
          "represents the layout of coordinate "
          "[xmin, ymin, xmax, ymax]."));
      PADDLE_ENFORCE_EQ(
        box_dims[1], score_dims[2],
        platform::errors::InvalidArgument(
          "The 2nd dimension of Input(BBoxes) must be equal to "
          "last dimension of Input(Scores), which represents the "
          "predicted bboxes."
          "But received box_dims[1](%s) != socre_dims[2](%s)",
          box_dims[1], score_dims[2]));
    }
    ctx->SetOutputDim("Out", {box_dims[1], box_dims[2] + 2});
    if (!ctx->IsRuntime()) {
      ctx->SetLoDLevel("Out", std::max(ctx->GetLoDLevel("BBoxes"), 1));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Scores"),
        platform::CPUPlace());
  }
};

template <typename T>
class MatrixNMSKernel : public framework::OpKernel<T> {
 public:
  void NMSMatrix(const Tensor& bbox, Tensor& scores,
                 const T score_threshold, const bool use_gaussian,
                 const T sigma, const int64_t top_k,
                 std::vector<int>* selected_indices,
                 const bool normalized) const {
    // The total boxes for each instance.
    int64_t num_boxes = bbox.dims()[0];
    int64_t box_size = bbox.dims()[1];

    std::vector<T> scores_data(num_boxes);
    std::copy_n(scores.data<T>(), num_boxes, scores_data.begin());
    std::vector<std::pair<T, int>> sorted_indices;
    GetMaxScoreIndex(scores_data, score_threshold, top_k, &sorted_indices);

    selected_indices->clear();
    const T* bbox_data = bbox.data<T>();

    auto num_pre = sorted_indices.size();
    std::vector<T> iou_matrix((num_pre * (num_pre - 1)) >> 1);
    std::vector<T> iou_max(num_pre);

    size_t i, j, ptr;
    T max_per_{0.};
    T min_per_{0.};
    int idx_a, idx_b;

    for (i = 0; i < num_pre; i++) {
      max_per_ = 0.;
      idx_a = sorted_indices[i].second;
      for (j = 0; j < i; j++) {
        idx_b = sorted_indices[j].second;
        auto iou = JaccardOverlap<T>(bbox_data + idx_a * box_size,
                                     bbox_data + idx_b * box_size, normalized);
        max_per_ = std::max(max_per_, iou);
        iou_matrix[ptr++] = iou;
      }
      iou_max[i] = max_per_;
    }

    // this snippet is duplicated because:
    // 1. branching in a hot loop could be slow
    // 2. code is short enough, not worth it to make a macro
    // 3. [[likely]] is only available since c++20 and compiler specific
    //    intrinsics is not desirable

    ptr = 0;
    T* scores_it = scores.data<T>();
    T decay;
    for (i = 0; i < num_pre; i++) {
      idx_a = sorted_indices[i].second;
      selected_indices->push_back(idx_a);
      min_per_ = 0.;

      for (j = 0; j < i; j++) {
        max_per_ = iou_max[j];
        auto iou = iou_matrix[ptr++];
        if (use_gaussian) {
          decay = std::exp(-(iou * iou - max_per_ * max_per_) / sigma);
        } else {
          decay = (1. - iou) / (1. - max_per_);
        }
        min_per_ = std::min(min_per_, decay);
      }
      *(scores_it + idx_a) *= min_per_;
    }
  }

  void MultiClassMatrixNMS(const framework::ExecutionContext& ctx,
                           const Tensor& scores_, const Tensor& bboxes,
                           std::map<int, std::vector<int>>* indices,
                           int* num_nmsed_out) const {
    int64_t background_label = ctx.Attr<int>("background_label");
    int64_t nms_top_k = ctx.Attr<int>("nms_top_k");
    int64_t keep_top_k = ctx.Attr<int>("keep_top_k");
    bool normalized = ctx.Attr<bool>("normalized");
    T score_threshold = static_cast<T>(ctx.Attr<float>("score_threshold"));
    auto& dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
    bool use_gaussian = ctx.Attr<bool>("use_gaussian");
    T gaussian_sigma = static_cast<T>(ctx.Attr<float>("gaussian_sigma"));
    int num_det = 0;

    Tensor scores;
    framework::TensorCopy(scores_, platform::CPUPlace(), dev_ctx, &scores);

    int64_t class_num = scores.dims()[0];
    Tensor bbox_slice, score_slice;
    int cls_det = 0;
    for (int64_t c = 0; c < class_num; ++c) {
      if (c == background_label) continue;
      score_slice = scores.Slice(c, c + 1);
      bbox_slice = bboxes;
      NMSMatrix(bbox_slice, score_slice, score_threshold,
                use_gaussian, gaussian_sigma, nms_top_k,
                &((*indices)[c]), normalized);
      cls_det = (*indices)[c].size();
      if (keep_top_k > -1 && cls_det > keep_top_k) {
        (*indices)[c].resize(keep_top_k);
      }
      num_det += cls_det;
    }

    *num_nmsed_out = num_det;
    const T* scores_data = scores.data<T>();
    if (keep_top_k > -1 && num_det > keep_top_k) {
      const T* sdata;
      std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
      for (const auto& it : *indices) {
        int label = it.first;
        sdata = scores_data + label * scores.dims()[1];
        const std::vector<int>& label_indices = it.second;
        for (size_t j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          score_index_pairs.push_back(
              std::make_pair(sdata[idx], std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::partial_sort(score_index_pairs.begin(),
                        score_index_pairs.begin() + keep_top_k,
                        score_index_pairs.end(),
                        SortScorePairDescend<std::pair<int, int>>);
      score_index_pairs.resize(keep_top_k);

      // Store the new indices.
      std::map<int, std::vector<int>> new_indices;
      for (size_t j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }
      new_indices.swap(*indices);
      *num_nmsed_out = keep_top_k;
    }
  }

  void MultiClassOutput(const platform::DeviceContext& ctx,
                        const Tensor& scores, const Tensor& bboxes,
                        const std::map<int, std::vector<int>>& selected_indices,
                        Tensor* outs, int* oindices = nullptr,
                        const int offset = 0) const {
    int64_t predict_dim = scores.dims()[1];
    int64_t box_size = bboxes.dims()[1];
    int64_t out_dim = box_size + 2;
    auto* scores_data = scores.data<T>();
    auto* bboxes_data = bboxes.data<T>();
    auto* odata = outs->data<T>();
    const T* sdata;
    Tensor bbox;
    bbox.Resize({scores.dims()[0], box_size});
    int count = 0;
    for (const auto& it : selected_indices) {
      int label = it.first;
      const std::vector<int>& indices = it.second;
      sdata = scores_data + label * predict_dim;
      for (size_t j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        odata[count * out_dim] = label;  // label
        const T* bdata;
        bdata = bboxes_data + idx * box_size;
        odata[count * out_dim + 1] = sdata[idx];  // score
        if (oindices != nullptr) {
          oindices[count] = offset + idx;
        }
        // xmin, ymin, xmax, ymax or multi-points coordinates
        std::memcpy(odata + count * out_dim + 2, bdata, box_size * sizeof(T));
        count++;
      }
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* boxes = ctx.Input<LoDTensor>("BBoxes");
    auto* scores = ctx.Input<LoDTensor>("Scores");
    auto* outs = ctx.Output<LoDTensor>("Out");
    auto score_dims = scores->dims();
    auto& dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();

    std::vector<std::map<int, std::vector<int>>> all_indices;
    std::vector<size_t> batch_starts = {0};
    int64_t batch_size = score_dims[0];
    int64_t box_dim = boxes->dims()[2];
    int64_t out_dim = box_dim + 2;
    int num_nmsed_out = 0;
    Tensor boxes_slice, scores_slice;
    for (int i = 0; i < batch_size; ++i) {
      scores_slice = scores->Slice(i, i + 1);
      scores_slice.Resize({score_dims[1], score_dims[2]});
      boxes_slice = boxes->Slice(i, i + 1);
      boxes_slice.Resize({score_dims[2], box_dim});
      std::map<int, std::vector<int>> indices;
      MultiClassMatrixNMS(ctx, scores_slice, boxes_slice, &indices,
                          &num_nmsed_out);
      all_indices.push_back(indices);
      batch_starts.push_back(batch_starts.back() + num_nmsed_out);
    }

    int num_kept = batch_starts.back();
    if (num_kept == 0) {
      T* od = outs->mutable_data<T>({1, 1}, ctx.GetPlace());
      od[0] = -1;
      batch_starts = {0, 1};
    } else {
      outs->mutable_data<T>({num_kept, out_dim}, ctx.GetPlace());
      int offset = 0;
      int* oindices = nullptr;
      for (int i = 0; i < batch_size; ++i) {
        scores_slice = scores->Slice(i, i + 1);
        boxes_slice = boxes->Slice(i, i + 1);
        scores_slice.Resize({score_dims[1], score_dims[2]});
        boxes_slice.Resize({score_dims[2], box_dim});
        int64_t s = batch_starts[i];
        int64_t e = batch_starts[i + 1];
        if (e > s) {
          Tensor out = outs->Slice(s, e);
          MultiClassOutput(dev_ctx, scores_slice, boxes_slice, all_indices[i],
                           &out, oindices, offset);
        }
      }
    }

    framework::LoD lod;
    lod.emplace_back(batch_starts);
    outs->set_lod(lod);
  }
};

class MatrixNMSOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("BBoxes",
             "(Tensor) A 3-D Tensor with shape "
             "[N, M, 4] represents the predicted locations of M bounding boxes"
             ", N is the batch size. "
             "Each bounding box has four coordinate values and the layout is "
             "[xmin, ymin, xmax, ymax], when box size equals to 4.");
    AddInput("Scores",
             "(Tensor) A 3-D Tensor with shape [N, C, M] represents the "
             "predicted confidence predictions. N is the batch size, C is the "
             "class number, M is number of bounding boxes. For each category "
             "there are total M scores which corresponding M bounding boxes. "
             " Please note, M is equal to the 2nd dimension of BBoxes. ");
    AddAttr<int>(
        "background_label",
        "(int, default: 0) "
        "The index of background label, the background label will be ignored. "
        "If set to -1, then all categories will be considered.")
        .SetDefault(0);
    AddAttr<float>("score_threshold",
                   "(float) "
                   "Threshold to filter out bounding boxes with low "
                   "confidence score. If not provided, consider all boxes.");
    AddAttr<int>("nms_top_k",
                 "(int64_t) "
                 "Maximum number of detections to be kept according to the "
                 "confidences after the filtering detections based on "
                 "score_threshold");
    AddAttr<int>("keep_top_k",
                 "(int64_t) "
                 "Number of total bboxes to be kept per image after NMS "
                 "step. -1 means keeping all bboxes after NMS step.");
    AddAttr<bool>("normalized",
                  "(bool, default true) "
                  "Whether detections are normalized.")
        .SetDefault(true);
    AddAttr<bool>("use_gaussian",
                  "(bool, default false) "
                  "Whether to use Gaussian as decreasing function.")
        .SetDefault(false);
    AddAttr<float>("gaussian_sigma",
                  "(float) "
                  "Sigma for Gaussian decreasing function, only takes effect ",
                   "when 'use_gaussian' is enabled.")
        .SetDefault(0.5);
    AddOutput("Out",
              "(LoDTensor) A 2-D LoDTensor with shape [No, 6] represents the "
              "detections. Each row has 6 values: "
              "[label, confidence, xmin, ymin, xmax, ymax]. "
              "the offsets in first dimension are called LoD, the number of "
              "offset is N + 1, if LoD[i + 1] - LoD[i] == 0, means there is "
              "no detected bbox.");
    AddComment(R"DOC(
This operator does multi-class matrix non maximum suppression (NMS) on batched
boxes and scores.
In the NMS step, this operator greedily selects a subset of detection bounding
boxes that have high scores larger than score_threshold, if providing this
threshold, then selects the largest nms_top_k confidences scores if nms_top_k
is larger than -1. Then this operator decays boxes score according to the
Matrix NMS scheme.
Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
per image if keep_top_k is larger than -1.
This operator support multi-class and batched inputs. It applying NMS
independently for each class. The outputs is a 2-D LoDTenosr, for each
image, the offsets in first dimension of LoDTensor are called LoD, the number
of offset is N + 1, where N is the batch size. If LoD[i + 1] - LoD[i] == 0,
means there is no detected bbox for this image.

For more information on Matrix NMS, please refer to:
https://arxiv.org/abs/2003.10152
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
  matrix_nms, ops::MatrixNMSOp, ops::MatrixNMSOpMaker,
  paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
  paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(matrix_nms, ops::MatrixNMSKernel<float>,
                       ops::MatrixNMSKernel<double>);
