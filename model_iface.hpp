/* ──────────────────────────────────────────────────────────────
   model_iface.hpp     –  the abstraction layer
   ────────────────────────────────────────────────────────────── */
#pragma once
#include "common.hpp"

struct TrainOpt {
    /* generic hyper-params – you may ignore the ones your model
       doesn’t need */
    int     trees       = 400;     // GBM / RF
    int     max_depth       = 10;      // tree depth or hidden1 size
    double  lr          = 0.06;
    double  subsample   = 0.7;
    double  colsample   = 0.85;
    bool    skip_train  = false;
    int     threads     = 0;
    int epochs = 10000;
    int patience=10;
    int hidden1=128;
};

struct IModel {
    virtual ~IModel() = default;

    /*  train **or** fine-tune on DS_tr, optionally eval on DS_val.
        The implementation decides whether to load existing file
        when opt.skip_train == true.                                */
    virtual void train(const std::vector<Sample>&    DS_tr,
                       const std::vector<Sample>&    DS_val,
                       const std::string&            model_path,
                       const TrainOpt&               opt) = 0;

    /*  hard 0/1 prediction on an arbitrary set                     */
    virtual std::vector<int>
        predict(const std::string&        model_path,
                const std::vector<Sample>& DS,
                float tau = 0.0f) const = 0;

    /*  helper so the framework can rank folds with BalAcc          */
    virtual double
        bal_acc (const std::string&        model_path,
                 const std::vector<Sample>& DS,
                 float tau = 0.0f) const = 0;
};
