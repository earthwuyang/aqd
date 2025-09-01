#pragma once
/*───────────────────────────────────────────────────────────
 *  vib_eigen.hpp  –  Variational-IB feature selector (Eigen)
 *───────────────────────────────────────────────────────────*/
#include "common.hpp"
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>


extern bool g_use_vib;                         // enabled by --vib
extern std::vector<int>       g_vib_idx;               // kept feature indices
extern std::vector<char>      g_feat_mask;             // 0/1 mask after VIB chosen


/* ===== 0. helpers ================================================= */
namespace vib {
using  M = Eigen::MatrixXf;
using  V = Eigen::VectorXf;

template<typename Derived>
inline auto relu(const Eigen::MatrixBase<Derived>& x)
        -> typename Derived::PlainObject { return x.cwiseMax(0.f); }

inline V sigmoid(const V& x){ return 1.f/(1.f+(-x.array()).exp()); }

inline V bce_with_logits(const V& l,const V& y){
    return (1.f-y.array())*l.array().exp().log1p()
         + y.array()      *(-l.array()).exp().log1p();
}

struct Adam{
    float lr,b1,b2,eps;  M m,v;
    Adam(float lr_=1e-3f):lr(lr_),b1(.9f),b2(.999f),eps(1e-8f){}
    void step(M& w,const M& g,int t){
        if(m.size()==0){ m=M::Zero(g.rows(),g.cols()); v=m; }
        m = b1*m + (1.f-b1)*g;
        v = b2*v + (1.f-b2)*g.cwiseProduct(g);
        float a = lr*std::sqrt(1.f-std::pow(b2,t))/(1.f-std::pow(b1,t));
        w.array() -= a * m.array() / (v.array().sqrt() + eps);
    }
};
} // namespace vib


/* ===== 1. very small MLP-VIB ====================================== */
struct VibNetEigen{
    using M=vib::M; using V=vib::V;
    int F,L; M W1,W2,Wc; V b1,b2,bc;

    VibNetEigen(int in_dim,int latent,std::mt19937& rng):F(in_dim),L(latent){
        std::normal_distribution<float> nd(0.f,0.02f);
        auto randm=[&](int r,int c){ M m(r,c);
            for(int i=0;i<r;++i)for(int j=0;j<c;++j) m(i,j)=nd(rng); return m;};
        auto randv=[&](int d){ return V::Zero(d); };
        W1=randm(128,F);   b1=randv(128);
        W2=randm(2*L,128); b2=randv(2*L);
        Wc=randm(1,L);     bc=randv(1);
    }

    struct Cache{ M x,z,h,mu,logvar,std; };

    std::pair<V,float> forward(const M& x,Cache& c,std::mt19937& rng){
        c.x = x;                                        // N×F
        c.h = vib::relu((W1*x.transpose()).colwise()+b1);         //128×N
        M z2 = (W2*c.h).colwise()+b2;                              //2L×N
        c.mu      = z2.topRows(L);
        c.logvar  = z2.bottomRows(L).cwiseMax(-10.f).cwiseMin(10.f);
        c.std     = (0.5f*c.logvar.array()).exp();

        std::normal_distribution<float> nd(0.f,1.f);
        M eps(L,x.rows()); for(int i=0;i<eps.size();++i) eps(i)=nd(rng);
        c.z = c.mu + c.std.cwiseProduct(eps);                      //L×N

        V logits = ((Wc*c.z).colwise()+bc).transpose();            //N
        V kl = -0.5f*(1.f+c.logvar.array()
                        -c.mu.array().square()
                        -c.logvar.array().exp()).colwise().sum();
        return {logits, kl.mean()};
    }

    void backward(const V& logits,const V& y,float beta,const Cache& c,
                  vib::Adam& o1,vib::Adam& o2,vib::Adam& o3,int step)
    {
        const int N = logits.size();
        V dL_dlog = vib::sigmoid(logits) - y;         // N
        M dL_dzc  = dL_dlog.transpose()/float(N);     // 1×N

        /* classifier grads */
        M gWc = dL_dzc * c.z.transpose();  V gbc=dL_dzc.rowwise().sum();
        M dL_dz = Wc.transpose()*dL_dzc;              // L×N

        /* KL grads */
        M dKL_dmu  =  c.mu / float(N);
        M dKL_dlogv= (0.5f*(c.logvar.array().exp()-1.f)).matrix()/float(N);
        M dL_dlogv = 0.5f*dL_dz.cwiseProduct(c.std);  // L×N

        dL_dz += beta*( dKL_dmu + dKL_dlogv.cwiseProduct(c.std) );

        /* split W2 -------------------------------------------------- */
        M topW = W2.topRows(L);     // for μ
        M botW = W2.bottomRows(L);  // for logσ²

        /* gradients wrt h */
        M dL_dh = topW.transpose()*dL_dz
                + botW.transpose()*dL_dlogv;

        /* ReLU grad */
        dL_dh = dL_dh.cwiseProduct( (c.h.array()>0.f).cast<float>().matrix() );

        /* W2 / b2 grads (μ part & logσ² part) */
        M gW2 = M::Zero(2*L,128);
        gW2.topRows(L)    = dL_dz    * c.h.transpose();
        gW2.bottomRows(L) = dL_dlogv * c.h.transpose();
        gW2.array() /= float(N);

        V gb2(2*L);
        gb2.topRows(L)    = dL_dz.rowwise().mean();
        gb2.bottomRows(L) = dL_dlogv.rowwise().mean();

        /* W1 / b1 grads */
        M gW1 = dL_dh * c.x;             V gb1 = dL_dh.rowwise().mean();

        /* Adam updates */
        o3.step(Wc,gWc,step);  bc.array() -= o3.lr*gbc.array();
        o2.step(W2,gW2,step);  b2.array() -= o2.lr*gb2.array();
        o1.step(W1,gW1,step);  b1.array() -= o1.lr*gb1.array();
    }
};


/* ===== 2. public entry =========================================== */
inline void vib_feature_mask_cpp(
        const std::vector<std::array<float,NUM_FEATS>>& X,
        const std::vector<int>&                        y,
        std::vector<float>&                            scores_out,
        int latent_dim=NUM_FEATS,int epochs=400,float beta=5e-5f,
        uint32_t seed=7,bool /*unused*/=false)
{
    const int N=int(X.size());
    if(N==0||y.size()!=X.size())
        throw std::runtime_error("vib: empty or mismatched input");

    vib::M Xt(N,NUM_FEATS);
    for(int i=0;i<N;++i) for(int j=0;j<NUM_FEATS;++j) Xt(i,j)=X[i][j];

    vib::V mu = Xt.colwise().mean();
    vib::V sd = ((Xt.rowwise()-mu.transpose()).array().square()
                 .colwise().mean()+1e-9f).sqrt();
    Xt = (Xt.rowwise()-mu.transpose()).array()
            .rowwise()/sd.transpose().array();

    vib::V yt(N); for(int i=0;i<N;++i) yt(i)=float(y[i]);

    std::mt19937 rng(seed);
    VibNetEigen net(NUM_FEATS,latent_dim,rng);
    vib::Adam o1(1e-4f),o2(1e-4f),o3(1e-4f);

    for(int e=1;e<=epochs;++e){
        VibNetEigen::Cache c;
        auto [logits,kl]=net.forward(Xt,c,rng);
        vib::V bce=vib::bce_with_logits(logits,yt);
        float loss=bce.mean()+beta*kl;
        net.backward(logits,yt,beta,c,o1,o2,o3,e);
        if(e%20==0) std::cout<<"[VIB] epoch "<<e<<'/'<<epochs
                             <<"  loss="<<loss<<'\n';
    }

    vib::V Wlat = net.W2.array().abs().matrix()
                      .colwise().mean().transpose();          //128
    vib::V scores=(net.W1.array().abs().matrix().transpose()*Wlat).array();

    scores = scores.cwiseProduct(sd)/scores.maxCoeff();
    scores /= scores.maxCoeff();
    scores_out.assign(scores.data(),scores.data()+NUM_FEATS);
}

/* 由挑出的下标生成 0/1 掩码 */
inline void build_feat_mask()
{
    g_feat_mask.assign(NUM_FEATS, 0);
    for (int id : g_vib_idx)
        if (id >= 0 && id < NUM_FEATS)
            g_feat_mask[id] = 1;
}