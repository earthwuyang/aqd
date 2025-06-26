/*  fannmlp_model.cpp  ---------------------------------------- */
#include "model_iface.hpp"
#include <fann.h>
#include <fann_cpp.h>
#if __cplusplus < 201402L   // 只有 C++14 之前的标准才进这里
#include <memory>
#include <utility>

namespace std {
    template <typename T, typename... Args>
    inline std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}
#endif

using fann_ptr = std::unique_ptr<struct fann, void(*)(struct fann*)>;
inline fann_ptr wrap_fann(struct fann *p) { return fann_ptr(p, fann_destroy); }

// /* 生成并训练一个三层 MLP： INPUT → h1 → h2 → 1  */
// static fann_ptr train_fann(const std::vector<float>& X,
//                            const std::vector<float>& y,
//                            unsigned nrow, unsigned nfeat,
//                            unsigned h1, unsigned h2,
//                            unsigned max_epoch   = 3000,
//                            float    lr          = 0.001f,
//                            float    desired_mse = 1e-4f)
// {
//     /* 1) 创建网络 (layer array 需要动态分配) */
//     const unsigned L = 4;                       // input + 2 hidden + output
//     std::vector<unsigned> layers = {nfeat, h1, h2, 1};
//     struct fann *net_raw = fann_create_standard_array(L, layers.data());
//     if(!net_raw) return wrap_fann(nullptr);

//     fann_set_activation_function_hidden(net_raw, FANN_SIGMOID_SYMMETRIC);
//     fann_set_activation_function_output(net_raw,  FANN_LINEAR);
//     fann_set_training_algorithm(net_raw, FANN_TRAIN_RPROP);
//     fann_set_learning_rate(net_raw, lr);

//     /* 2) 生成训练数据 */
//     struct fann_train_data *td = fann_create_train(nrow, nfeat, 1);
//     for (unsigned i=0;i<nrow;++i) {
//         memcpy(td->input[i],  &X[i*nfeat], nfeat*sizeof(float));
//         td->output[i][0] = y[i];
//     }

//     /* 3) 训练 */
//     fann_train_on_data(net_raw, td, max_epoch, 100, desired_mse);
//     fann_destroy_train(td);
//     return wrap_fann(net_raw);
// }

/* 单样本推理 */
inline float fann_predict(const struct fann *net, const float *feat)
{
    // fann_run 需要可写指针，做 const_cast
    return fann_run(const_cast<fann*>(net),
                    const_cast<float*>(feat))[0];
}

/* 保存网络；返回 true=成功 */
inline bool fann_save_net(const struct fann *net, const std::string& path)
{
    return fann_save(const_cast<fann*>(net), path.c_str()) == 0;
}



class FANNModel : public IModel {
   /* ---------------------------------------------------------------
   *  FANNModel::train – copy-paste this whole definition verbatim
   * --------------------------------------------------------------- */
   void train(const std::vector<Sample>& DS,
            const std::vector<Sample>& va,
            const std::string&        model_path,
            const TrainOpt&           opt) override
   {
      /* ---------- hyper-parameters from CLI / TrainOpt ------------ */
      const bool   skip_train = opt.skip_train;
      const int    max_epoch  = opt.epochs   > 0 ? opt.epochs   : 10000;
      const int    patience   = opt.patience > 0 ? opt.patience : 30;
      const float  lr         = float(opt.lr > 0 ? opt.lr : 0.05);
      const int    h1_size    = opt.hidden1;   // hidden-layer 1
      const int    h2_size    = h1_size / 2;

      constexpr float  TAU_STAR = 0.0f;                       // ŷ>0 ⇒ choose column

      /* ---------- 1. build TRAIN matrices ------------------------- */
      std::vector<const Sample*> S;
      for (const auto& s : DS)
         if (s.row_t > 0 && s.col_t > 0)
               S.push_back(&s);

      const unsigned N = static_cast<unsigned>(S.size());
      if (!N) { logE("no usable samples"); return; }

      std::vector<float> X(size_t(N) * NUM_FEATS);
      std::vector<float> Y(N);                          // regression targets

      for (unsigned i = 0; i < N; ++i) {
         const Sample& s = *S[i];
         std::memcpy(&X[i * NUM_FEATS], s.feat.data(),
                     NUM_FEATS * sizeof(float));
         Y[i] = std::log(std::max(s.row_t, EPS_RUNTIME)) -
                  std::log(std::max(s.col_t, EPS_RUNTIME));
      }

      /* ---------- 2. optional VALID matrices ---------------------- */
      std::vector<float> X_val, Y_val;
      unsigned Nv = static_cast<unsigned>(va.size());
      if (Nv) {
         X_val.resize(size_t(Nv) * NUM_FEATS);
         Y_val.resize(Nv);
         for (unsigned i = 0; i < Nv; ++i) {
               const Sample& s = va[i];
               std::memcpy(&X_val[i * NUM_FEATS], s.feat.data(),
                           NUM_FEATS * sizeof(float));
               Y_val[i] = std::log(std::max(s.row_t, EPS_RUNTIME)) -
                        std::log(std::max(s.col_t, EPS_RUNTIME));
         }
      }

      /* ---------- 3. (maybe) TRAIN the network -------------------- */
      fann_ptr net(nullptr, nullptr);                   // smart-ptr wrapper

      /* ---- 3-a. inference-only branch ---------------------------- */
      if (skip_train) {
         net = wrap_fann(fann_create_from_file(model_path.c_str()));
         if (!net) { logE("model load failed"); return; }
      }
      /* ---- 3-b. full training with patience ---------------------- */
      else {
         /* create:  INPUT → h1 → h2 → OUTPUT(1) */
         net = wrap_fann(fann_create_standard(
               4,                   // 4 layers total
               NUM_FEATS,
               static_cast<unsigned>(h1_size),
               static_cast<unsigned>(h2_size),
               1));
         if (!net) { logE("FANN create failed"); return; }

         fann_set_training_algorithm(net.get(), FANN_TRAIN_RPROP);
         fann_set_learning_rate      (net.get(), lr);
         fann_set_activation_function_hidden(net.get(), FANN_SIGMOID_SYMMETRIC);
         fann_set_activation_function_output(net.get(), FANN_LINEAR);

         /* ---- build fann_train_data for TRAIN ------------------- */
         std::vector<fann_type*> train_in (N), train_out(N);
         for (unsigned i = 0; i < N; ++i) {
               train_in [i] = &X[i * NUM_FEATS];
               train_out[i] = &Y[i];                    // scalar target
         }
         struct fann_train_data tr;
         tr.num_data   = N;
         tr.num_input  = NUM_FEATS;
         tr.num_output = 1;
         tr.input      = train_in .data();           // float** → fann_type**
         tr.output     = train_out.data();

         /* ---- idem for VALID (if provided) ---------------------- */
         struct fann_train_data val = {};
         std::vector<fann_type*> val_in, val_out;
         if (Nv) {
               val_in .resize(Nv);
               val_out.resize(Nv);
               for (unsigned i = 0; i < Nv; ++i) {
                  val_in [i] = &X_val[i * NUM_FEATS];
                  val_out[i] = &Y_val[i];
               }
               val.num_data   = Nv;
               val.num_input  = NUM_FEATS;
               val.num_output = 1;
               val.input      = val_in .data();
               val.output     = val_out.data();
         }

         /* helper: MSE of a fann_train_data block ----------------- */
         auto mse_on = [&](const struct fann_train_data& d)->float {
               double mse = 0.0;
               for (unsigned i = 0; i < d.num_data; ++i) {
                  float *out = fann_run(net.get(),
                                       const_cast<float*>(d.input[i]));
                  float diff = out[0] - d.output[i][0];
                  mse += diff * diff;
               }
               return static_cast<float>(mse / std::max(1u, d.num_data));
         };

         /* ---- epoch loop with early-stopping -------------------- */
         float best_mse = std::numeric_limits<float>::max();
         int   best_ep  = -1;
         int   stall    = 0;

         for (int ep = 0; ep < max_epoch; ++ep) {
               fann_train_epoch(net.get(), &tr);

               float cur_mse = Nv ? mse_on(val) : fann_get_MSE(net.get());

               if (cur_mse < best_mse - 1e-6f) {       // improved
                  best_mse = cur_mse;
                  best_ep  = ep;
                  stall    = 0;
                  fann_save(net.get(), model_path.c_str());   // checkpoint
               } else if (++stall >= patience) {
                  logI("Early-stop at epoch " + std::to_string(ep + 1) +
                        "  best=" + std::to_string(best_ep + 1) +
                        "  val_MSE=" + std::to_string(best_mse));
                  break;
               }

               if ((ep + 1) % 1 == 0 || ep == 0)
                  progress("epoch", ep + 1, max_epoch);
         }

         /* reload best checkpoint -------------------------------- */
         net = wrap_fann(fann_create_from_file(model_path.c_str()));
         logI("Model saved → " + model_path);
      }

      /* ---------- 4. quick evaluation on DS ----------------------- */
      long   TP=0, FP=0, TN=0, FN=0;
      double r_row=0, r_col=0, r_opt=0,
            r_costthr=0, r_hyp=0, r_fann_imci=0, r_fannmlp=0;

      for (const auto& s : DS) {
         float yhat      = fann_predict(net.get(), s.feat.data());
         bool  col_mlpm  = (yhat > TAU_STAR);          // 本模型
         bool  col_hyp   = (s.hybrid_pred == 1);       // Hybrid Optimizer
         bool  col_fann0 = (s.fann_pred   == 1);       // 旧 FANN 基线
         bool  use_row   = (s.qcost < COST_THR);       // cost-threshold

         /* ----------- runtime 累加 ---------------- */
         r_row       += s.row_t;
         r_col       += s.col_t;
         r_opt       += std::min(s.row_t, s.col_t);
         r_costthr   += use_row   ? s.row_t : s.col_t;
         r_hyp       += col_hyp   ? s.col_t : s.row_t;
         r_fann_imci += col_fann0 ? s.col_t : s.row_t;
         r_fannmlp   += col_mlpm  ? s.col_t : s.row_t;

         /* ----------- 仅对 FANN-MLP 记准确率 ------- */
         (col_mlpm ? (s.label?++TP:++FP) : (s.label?++FN:++TN));
      }

      auto avg = [&](double v){ return v / DS.size(); };
      std::cout << std::fixed << std::setprecision(6)
               << "Acc="     << double(TP+TN)/DS.size()
               << "  BalAcc="<< 0.5*(double(TP)/(TP+FN)+double(TN)/(TN+FP))
               << "  F1="     << (TP?2.0*TP/(2*TP+FP+FN):0) << '\n'
               << "Row only        : " << avg(r_row)        << '\n'
               << "Col only        : " << avg(r_col)        << '\n'
               << "Cost-Threshold  : " << avg(r_costthr)    << '\n'
               << "HybridOpt       : " << avg(r_hyp)        << '\n'
               << "FANN baseline   : " << avg(r_fann_imci)  << '\n'
               << "FANN-MLP (this) : " << avg(r_fannmlp)    << '\n'
               << "Oracle (min)    : " << avg(r_opt)        << "\n\n";
   }



   std::vector<int> predict(const std::string& model_path,
                             const std::vector<Sample>& DS,
                             float tau) const override
   {
      const int N = DS.size();
      vector<int> bin(N, 0);

      /* 1) 载入 .net 文件 */
      struct fann *net = fann_create_from_file(model_path.c_str());
      if (!net) {
         logE("load " + model_path + " failed");
         exit(1);
      }
      if (fann_get_num_input(net) != NUM_FEATS) {
         logE("feature-dim mismatch in " + model_path);
         fann_destroy(net);
         exit(1);
      }

      /* 2) 逐行推理 */
      for (int i = 0; i < N; ++i) {
         /* array<float,NUM_FEATS> → float*  (FANN 要求非 const) */
         float *out = fann_run(net,
                                 const_cast<float*>(DS[i].feat.data()));
         bin[i] = (out[0] > tau);          // ŷ>τ ⇒ 预测列存
      }

      fann_destroy(net);
      return bin;
   }




   double bal_acc(const std::string& model_path,
                   const std::vector<Sample>& DS,
                   float tau) const override
    {
      if (DS.empty()) return 0.0;

      /* 1) 调用 FANN 版 predict_binary() 拿到列/行二分类结果 */
      std::vector<int> pred = predict(model_path, DS, tau);
      const int N = static_cast<int>(pred.size());

      /* 2) 统计混淆矩阵 */
      int TP = 0, TN = 0, FP = 0, FN = 0;
      for (int i = 0; i < N; ++i) {
         bool col = pred[i];          // 1 ⇒ 预测列存
         bool gt  = DS[i].label;      // 1 ⇒ 列存更快 (真实标签)

         if      ( col &&  gt) ++TP;
         else if ( col && !gt) ++FP;
         else if (!col &&  gt) ++FN;
         else                   ++TN;
      }

      /* 3) 计算 Balanced Accuracy */
      double tpr = (TP + FN) ? static_cast<double>(TP) / (TP + FN) : 0.0;
      double tnr = (TN + FP) ? static_cast<double>(TN) / (TN + FP) : 0.0;
      return 0.5 * (tpr + tnr);
    }
};

std::unique_ptr<IModel> make_fann() { return std::make_unique<FANNModel>(); }
