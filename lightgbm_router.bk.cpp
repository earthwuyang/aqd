/*****************************************************************************************
 * lightgbm_router.cpp  –  63-feature LightGBM regressor (row-vs-column decision)
 *            ♦ portable 〈filesystem〉 shim
 *            ♦ identical feature extraction as train_dtree_enhanced.cpp
 *            ♦ LightGBM C-API (“regression”) → choose column when ŷ > 0
 *
 * Build (example):
 *   g++ -O3 -std=c++17 -I$LIGHTGBM/include \
 *       lightgbm_router.cpp -o lightgbm_router \
 *       -L$LIGHTGBM/lib -l_lightgbm -lmysqlclient -pthread -lstdc++fs
 *
 * Run:
 *   ./lightgbm_router --data_dirs=airline,hepatitis --base=/home/wuy/query_costs
 *****************************************************************************************/
#include <bits/stdc++.h>

/* ---------- portable <filesystem> ---------- */
#if __has_include(<filesystem>)
    #include <filesystem>
    namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#else
    #error "Your compiler/libstdc++ lacks <filesystem>"
#endif
/* ------------------------------------------- */

#include <regex>
#include <mysql/mysql.h>
#include "json.hpp"
#include <LightGBM/c_api.h>

using json = nlohmann::json;
using namespace std;

/* ───────────────── helpers ───────────────── */
constexpr int    NUM_FEATS   = 63;
constexpr double EPS_RUNTIME = 1e-6;

static void logI(const string&s){ cerr<<"[INFO]  "<<s<<'\n'; }
static void logW(const string&s){ cerr<<"[WARN]  "<<s<<'\n'; }
static void logE(const string&s){ cerr<<"[ERR]   "<<s<<'\n'; }

static void progress(const string&tag,size_t cur,size_t tot,size_t W=40){
    double f=tot?double(cur)/tot:1.0; size_t filled=size_t(f*W);
    cerr<<"\r"<<tag<<" ["<<string(filled,'=')<<string(W-filled,' ')
        <<"] "<<setw(3)<<int(f*100)<<"% ("<<cur<<'/'<<tot<<')'<<flush;
    if(cur==tot) cerr<<'\n';
}

static double safe_f(const json&v){
    if(v.is_number()) return v.get<double>();
    if(v.is_string()){ try{ return stod(v.get<string>());}catch(...){ } }
    return 0.0;
}
static double safe_f(const json&o,const char*k){ return o.contains(k)?safe_f(o[k]):0; }
static double log1p_clip(double v){ return log1p(max(0.0,v)); }
static double str_size_to_num(string s){
    if(s.empty()) return 0.0;
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    double m=1; char suf=s.back();
    if(suf=='G'||suf=='g'){m=1e9; s.pop_back();}
    else if(suf=='M'||suf=='m'){m=1e6; s.pop_back();}
    else if(suf=='K'||suf=='k'){m=1e3; s.pop_back();}
    try{ return stod(s)*m;}catch(...){ return 0.0;}
}
static bool getBool(const json&o,const char*k){
    if(!o.contains(k)) return false;
    const auto&v=o[k];
    if(v.is_boolean()) return v.get<bool>();
    if(v.is_string()){ string s=v.get<string>(); transform(s.begin(),s.end(),s.begin(),::tolower);
                       return s=="yes"||s=="true"||s=="1";}
    return false;
}

/* ─────────── MySQL index-map (unchanged logic) ─────────── */
static unordered_map<string, unordered_set<string>> indexCols;

static void load_index_defs_from_db(const string &host,int port,
                                    const string &user,const string &pass,
                                    const string &db,const string &tbl)
{
    MYSQL *conn = mysql_init(nullptr);
    if(!mysql_real_connect(conn,host.c_str(),user.c_str(),pass.c_str(),
                           db.c_str(),port,nullptr,0))
    { logW("MySQL connect failed: "+string(mysql_error(conn))); return; }

    string q="SHOW CREATE TABLE `"+tbl+"`";
    if(mysql_query(conn,q.c_str()))
    { logW("SHOW CREATE TABLE failed: "+string(mysql_error(conn))); mysql_close(conn); return; }

    MYSQL_RES *res=mysql_store_result(conn);
    if(!res){ mysql_close(conn); return; }
    MYSQL_ROW row=mysql_fetch_row(res);
    if(!row||!row[1]){ mysql_free_result(res); mysql_close(conn); return; }
    string ddl=row[1]; mysql_free_result(res); mysql_close(conn);

    static const regex re(R"(KEY\s+`([^`]+)`\s*\(\s*([^)]+)\))",regex::icase);
    smatch m; auto it=ddl.cbegin(), ed=ddl.cend();
    while(regex_search(it,ed,m,re)){
        string idx=m[1], colsRaw=m[2];
        unordered_set<string> S;
        string col; stringstream ss(colsRaw);
        while(getline(ss,col,',')){
            col.erase(remove(col.begin(),col.end(),'`'),col.end());
            col.erase(0,col.find_first_not_of(" \t"));
            col.erase(col.find_last_not_of(" \t")+1);
            if(!col.empty()) S.insert(col);
        }
        indexCols[idx]=std::move(S);
        it=m.suffix().first;
    }
}

static void load_all_index_defs(const string &host,int port,
                                const string &user,const string &pass,
                                const vector<string>& dbs)
{
    indexCols.clear();
    for(const auto&db:dbs){
        MYSQL *c=mysql_init(nullptr);
        if(!mysql_real_connect(c,host.c_str(),user.c_str(),pass.c_str(),
                               db.c_str(),port,nullptr,0))
        { logW("connect "+db+" failed"); mysql_close(c); continue; }

        if(mysql_query(c,"SHOW TABLES"))
        { logW("SHOW TABLES failed in "+db); mysql_close(c); continue; }
        MYSQL_RES *t=mysql_store_result(c);
        MYSQL_ROW r;
        while((r=mysql_fetch_row(t))) load_index_defs_from_db(host,port,user,pass,db,r[0]);
        mysql_free_result(t); mysql_close(c);
    }
}
/* -------------------------------------------------------------------- */


/* ─────────── 63-feature extraction ─────────── */
struct Agg{
    double re=0,rp=0,f=0,rc=0,ec=0,pc=0,dr=0,selSum=0,selMin=1e30,selMax=0,
           ratioSum=0,ratioMax=0,maxPrefix=0,minRead=1e30,fanoutMax=0;
    int    cnt=0,cRange=0,cRef=0,cEq=0,cIdx=0,cFull=0,idxUse=0,sumPK=0,
           coverCount=0,maxDepth=0;
    bool   grp=false,ord=false,tmp=false;
};
static void walk(const json& n,Agg&a,int depth=1){
    if(n.is_object()){
        if(n.contains("table")&&n["table"].is_object()){
            const auto&t=n["table"]; const auto&ci=t.value("cost_info",json::object());
            double re=safe_f(t,"rows_examined_per_scan");
            double rp=safe_f(t,"rows_produced_per_join");
            double fl=safe_f(t,"filtered");
            double rc=safe_f(ci,"read_cost");
            double ec=safe_f(ci,"eval_cost");
            double pc=safe_f(ci,"prefix_cost");
            double dr=ci.contains("data_read_per_join")&&ci["data_read_per_join"].is_string()
                      ? str_size_to_num(ci["data_read_per_join"].get<string>())
                      : safe_f(ci,"data_read_per_join");
            a.re+=re; a.rp+=rp; a.f+=fl; a.rc+=rc; a.ec+=ec; a.pc+=pc; a.dr+=dr; a.cnt++;
            a.maxPrefix=max(a.maxPrefix,pc); a.minRead=min(a.minRead,rc);
            if(re>0){ double sel=rp/re; a.selSum+=sel; a.selMin=min(a.selMin,sel); a.selMax=max(a.selMax,sel); a.fanoutMax=max(a.fanoutMax,sel); }
            double ratio=ec>0?rc/ec:rc; a.ratioSum+=ratio; a.ratioMax=max(a.ratioMax,ratio);
            string at=t.value("access_type","ALL");
            if(at=="range")a.cRange++; else if(at=="ref")a.cRef++; else if(at=="eq_ref")a.cEq++; else if(at=="index")a.cIdx++; else a.cFull++;
            if(getBool(t,"using_index")) a.idxUse++;
            if(t.contains("possible_keys")&&t["possible_keys"].is_array()) a.sumPK+=int(t["possible_keys"].size());
            /* covering index check */
            if(t.contains("used_columns")&&t["used_columns"].is_array()&&t.contains("key")&&t["key"].is_string()){
                string idx=t["key"]; auto it=indexCols.find(idx);
                if(it!=indexCols.end()){
                    bool cover=true;
                    for(auto&u:t["used_columns"])
                        if(!u.is_string()||!it->second.count(u.get<string>())){cover=false;break;}
                    if(cover) a.coverCount++;
                }
            }
        }
        if(n.contains("grouping_operation")) a.grp=1;
        if(n.contains("ordering_operation")||getBool(n,"using_filesort")) a.ord=1;
        if(getBool(n,"using_temporary_table")) a.tmp=1;
        for(auto&kv:n.items()) if(kv.key()!="table") walk(kv.value(),a,depth+1);
    }else if(n.is_array()) for(auto&v:n) walk(v,a,depth);
    a.maxDepth=max(a.maxDepth,depth);
}

static bool plan2feat(const json&plan,float f[NUM_FEATS]){
    if(!plan.contains("query_block")) return false;
    const json*qb=&plan["query_block"];
    if(qb->contains("union_result")){
        const auto& specs=(*qb)["union_result"]["query_specifications"];
        if(specs.is_array()&&!specs.empty()) qb=&specs[0]["query_block"];
    }
    Agg a; walk(*qb,a);
    if(!a.cnt) return false;
    double inv=1.0/a.cnt; int k=0;
    double qCost=safe_f(qb->value("cost_info",json::object()),"query_cost");
    double rootRow=safe_f(*qb,"rows_produced_per_join");
#define PUSH(x) f[k++]=float(x)
    /* 0-6 basic */           PUSH(log1p_clip(a.re*inv));  PUSH(log1p_clip(a.rp*inv)); PUSH(log1p_clip(a.f*inv));
    PUSH(log1p_clip(a.rc*inv)); PUSH(log1p_clip(a.ec*inv)); PUSH(log1p_clip(a.pc*inv)); PUSH(log1p_clip(a.dr*inv));
    /* 7-12 access */         PUSH(a.cRange*inv); PUSH(a.cRef*inv); PUSH(a.cEq*inv); PUSH(a.cIdx*inv); PUSH(a.cFull*inv); PUSH(a.idxUse*inv);
    /* 13-17 sel/shape */     PUSH(a.selSum*inv); PUSH(a.selMin); PUSH(a.selMax); PUSH(a.maxDepth); PUSH(a.fanoutMax);
    /* 18-20 flags */         PUSH(a.grp); PUSH(a.ord); PUSH(a.tmp);
    /* 21-22 ratios */        PUSH(a.ratioSum*inv); PUSH(a.ratioMax);
    /* 23-24 cost/rows */     PUSH(log1p_clip(qCost)); PUSH(log1p_clip(rootRow));
    /* 25-27 cost ratios */   PUSH(log1p_clip((a.pc*inv)/max(1e-6,a.rc*inv)));
    PUSH(log1p_clip((a.rc*inv)/max(1e-6,a.re*inv)));
    PUSH(log1p_clip((a.ec*inv)/max(1e-6,a.re*inv)));
    /* 28-31 */               PUSH(a.cnt==1); PUSH(a.cnt>1);
    PUSH(log1p_clip(a.maxDepth*(a.idxUse*inv)));
    PUSH(log1p_clip((a.idxUse*inv)/max(a.cFull*inv,1e-3)));
    /* 32-39 misc */          PUSH(a.cnt); PUSH(a.cnt?double(a.sumPK)/a.cnt:0);
    PUSH(log1p_clip(a.maxPrefix)); PUSH(log1p_clip(a.minRead<1e30?a.minRead:0));
    PUSH(a.cnt>1?double(a.cnt-1)/a.cnt:0); PUSH(rootRow>0?double(a.re)/rootRow:0);
    PUSH(a.selMax-a.selMin); PUSH(a.idxUse/double(max(1,a.cRange+a.cRef+a.cEq+a.cIdx)));
    /* 40-43 */               PUSH(qCost); PUSH(qCost>5e4); PUSH(a.cnt?double(a.coverCount)/a.cnt:0); PUSH(a.coverCount==a.cnt);
    /* 44-46 */               PUSH(log1p_clip(a.re*inv)-log1p_clip(a.selSum*inv)); PUSH(a.cnt); PUSH(log1p_clip(a.cnt));
    /* 47-50 */               PUSH(a.sumPK); PUSH(a.cnt?double(a.sumPK)/a.cnt:0); PUSH(a.coverCount); PUSH(a.cnt?double(a.coverCount)/a.cnt:0);
    /* 51-56 */               PUSH(a.idxUse*inv); PUSH(a.cRange*inv); PUSH(a.cRef*inv); PUSH(a.cEq*inv); PUSH(a.cIdx*inv); PUSH(a.cFull*inv);
    /* 57-59 */               PUSH(log1p_clip(a.maxPrefix*inv)); PUSH(log1p_clip(a.minRead<1e30?a.minRead:0)); PUSH(a.selMax-a.selMin);
    /* 60-62 extremes */      PUSH(a.ratioMax); PUSH(a.fanoutMax); PUSH(a.selMin>0?double(a.selMax/a.selMin):0);
    return k==NUM_FEATS;
#undef PUSH
}

/* ─────────── sample struct & loaders ─────────── */
struct Sample{
    array<float,NUM_FEATS> feat{};
    int    label = 0;              // 真实最优: 1=column 更快, 0=row 更快
    double row_t = 0, col_t = 0;   // 两种执行路径真实耗时 (s)
    double qcost = 0;              // Optimizer 估计的 query_cost
    int    fann_pred   = -1;       // 0/1；若 CSV 中缺失则保持 -1
    int    hybrid_pred = -1;       // 0/1；若 CSV 中缺失则保持 -1
};

static bool load_plan_file(const string&path,array<float,NUM_FEATS>&f,double&q){
    ifstream in(path); if(!in) return false; json j; try{ in>>j;}catch(...){return false;}
    if(!plan2feat(j,f.data())) return false;
    q= j.contains("query_block") ? safe_f(j["query_block"].value("cost_info",json::object()),"query_cost") : 0;
    return true;
}

static vector<Sample> load_dataset(const string&root,
                                   const vector<string>&dirs)
{
    vector<Sample> DS;

    for(const auto& d : dirs){
        string csv = root + "/" + d + "/query_costs.csv";
        fs::path planDir = root + "/" + d + "/row_plans";
        if(!fs::exists(csv) || !fs::is_directory(planDir)){
            logW("skip "+d); continue;
        }

        /* ---------- 读 meta ---------- */
        struct MetaRec{ int lab; double rt, ct; int fann, hybrid; };
        unordered_map<string,MetaRec> meta;

        ifstream fin(csv); string line;
        getline(fin,line);                     // 丢掉表头
        while(getline(fin,line)){
            stringstream ss(line);
            string qid,lab,rt,ct,fann,hyb;
            getline(ss,qid,','); getline(ss,lab,',');
            getline(ss,rt,',');  getline(ss,ct,',');
            getline(ss,hyb,',');getline(ss,fann,',');

            meta[qid] = {
                lab=="1",
                rt.empty()?0:stod(rt),
                ct.empty()?0:stod(ct),
                fann.empty()? -1 : stoi(fann),
                hyb .empty()? -1 : stoi(hyb )
            };
        }

        /* ---------- 扫 plan ---------- */
        vector<fs::path> files;
        for(auto&p:fs::directory_iterator(planDir))
            if(p.path().extension()==".json") files.push_back(p.path());

        size_t cur = 0, tot = files.size();
        for(auto& pf : files){
            ++cur; if(cur%5000==0) progress("scan "+d,cur,tot);

            string qid = pf.stem().string();
            auto it = meta.find(qid);
            if(it == meta.end()) continue;

            Sample s;
            if(!load_plan_file(pf.string(),s.feat,s.qcost)) continue;
            s.label      = it->second.lab;
            s.row_t      = it->second.rt;
            s.col_t      = it->second.ct;
            s.fann_pred  = it->second.fann;
            s.hybrid_pred= it->second.hybrid;
            DS.push_back(std::move(s));
        }
    }

    logI("Loaded "+to_string(DS.size())+" samples");
    return DS;
}



static void train_and_eval(const std::vector<Sample>& DS,
                           const std::string& model_path,
                           int     num_trees      = 400,
                           int     max_depth      = 10,
                           double  lr             = 0.06,
                           double  subsample      = 0.7,
                           double  colsample      = 0.8,
                           bool    skip_train     = false,
                           int     num_threads    = 100   /* 0 ⇒ auto */ )
{
    /* ---------- sanity ---------- */
    const int N = static_cast<int>(DS.size());
    if(!N){ logE("empty dataset"); return; }
    if(num_threads<=0) num_threads = max(1u,thread::hardware_concurrency());

    /* ---------- helper ---------- */
    auto foz=[](double v){ return isfinite(v)?v:0.0; };

    /* ---------- build arrays ---------- */
    vector<float>  X(size_t(N)*NUM_FEATS), y(N), w(N);
    constexpr double COST_DEN = 1e5;
    double P=0,N0=0;
    for(auto const&s:DS) s.label?++P:++N0;
    double w_pos = P  ? N/(2*P)  : 1.0;
    double w_neg = N0 ? N/(2*N0) : 1.0;

    for(int i=0;i<N;++i){
        for(int j=0;j<NUM_FEATS;++j)
            X[i*NUM_FEATS+j] = static_cast<float>(foz(DS[i].feat[j]));

        double rt=max(DS[i].row_t,EPS_RUNTIME);
        double ct=max(DS[i].col_t,EPS_RUNTIME);
        y[i]=float(log(rt)-log(ct));

        double gap=fabs(rt-ct);
        double w_gap=1+log1p(gap);
        double w_cost=1+min(DS[i].qcost,1e6)/COST_DEN;
        w[i]=float((DS[i].label?w_pos:w_neg)*w_gap*w_cost);
    }

    /* ---------- build / load booster ---------- */
    BoosterHandle booster=nullptr;
    if(skip_train){
        int dummy_iters=0;
        if(LGBM_BoosterCreateFromModelfile(model_path.c_str(),
                                           &dummy_iters,&booster)!=0||!booster){
            logE("model load failed: "+model_path); return;
        }
        logI("Loaded model ← "+model_path+"  ("+to_string(dummy_iters)+" trees)");
    }else{
        DatasetHandle dtrain;
        if(LGBM_DatasetCreateFromMat(
               X.data(),C_API_DTYPE_FLOAT32,
               N,NUM_FEATS,1,"",nullptr,&dtrain)){
            logE("DatasetCreate failed"); return;
        }
        LGBM_DatasetSetField(dtrain,"label", y.data(),N,C_API_DTYPE_FLOAT32);
        LGBM_DatasetSetField(dtrain,"weight",w.data(),N,C_API_DTYPE_FLOAT32);

        string param=
             "objective=regression metric=l2"
             " max_bin=127 num_leaves=64"
             " num_iterations="+to_string(num_trees)+
             " learning_rate="+to_string(lr)+
             " max_depth="+to_string(max_depth)+
             " feature_fraction="+to_string(colsample)+
             " bagging_fraction="+to_string(subsample)+
             " bagging_freq=1"+
             " num_threads="+to_string(num_threads)+
             " verbosity=-1";

        if(LGBM_BoosterCreate(dtrain,param.c_str(),&booster)){
            logE("BoosterCreate failed"); return;
        }

        for(int it=0,finish;it<num_trees;++it){
            LGBM_BoosterUpdateOneIter(booster,&finish);
            if((it+1)%10==0||it+1==num_trees) progress("train",it+1,num_trees);
        }
        cerr<<"\n";
        LGBM_BoosterSaveModel(booster,0,-1,0,model_path.c_str());
        logI("Model saved → "+model_path);
        LGBM_DatasetFree(dtrain);
    }

    /* ---------- prediction ---------- */
    vector<double> pred(N); int64_t out_len=0;
    if(LGBM_BoosterPredictForMat(
           booster,X.data(),C_API_DTYPE_FLOAT32,
           N,NUM_FEATS,1,C_API_PREDICT_NORMAL,
           -1,0,"",&out_len,pred.data())){
        logE("PredictForMat failed"); LGBM_BoosterFree(booster); return;
    }

    /* ---------- metrics ---------- */
    constexpr double COST_THR = 5e4;
    int TP=0,FP=0,TN=0,FN=0;
    double r_row=0,r_col=0,r_hopt=0,r_lgb=0,r_opt=0,r_fann=0;
    size_t n_fann_avail=0, n_hopt_avail=0;

    for(int i=0;i<N;++i){
        bool lgb_col = pred[i]>0.0;
        (lgb_col ? (DS[i].label?++TP:++FP)
                 : (DS[i].label?++FN:++TN));

        /* 基线 & 真实最优 */
        r_row += DS[i].row_t;
        r_col += DS[i].col_t;
        r_opt += min(DS[i].row_t,DS[i].col_t);

        /* hybrid optimizer (若 CSV 给出 hybrid_pred，否则用简单 cost-threshold 规则) */
        bool h_col;
        if(DS[i].hybrid_pred!=-1){
            h_col = DS[i].hybrid_pred;
        }else{
            h_col = (DS[i].qcost > COST_THR);          // 旧 cost rule
        }
        r_hopt += h_col ? DS[i].col_t : DS[i].row_t;
        ++n_hopt_avail;

        /* FANN 预测 (仅当存在时) */
        if(DS[i].fann_pred!=-1){
            bool f_col = DS[i].fann_pred;
            r_fann += f_col ? DS[i].col_t : DS[i].row_t;
            ++n_fann_avail;
        }

        /* LightGBM */
        r_lgb += lgb_col ? DS[i].col_t : DS[i].row_t;
    }

    auto avg=[&](double sum,size_t n){ return n?sum/n:std::numeric_limits<double>::quiet_NaN(); };
    cout << fixed << setprecision(6)
         << "\n=== CONFUSION (LightGBM) ===\nTP="<<TP<<" FP="<<FP
         << " TN="<<TN<<" FN="<<FN
         << "\nAccuracy="<<double(TP+TN)/N
         << "  BalAcc="<<0.5*(double(TP)/(TP+FN)+double(TN)/(TN+FP))
         << "  F1="<<((TP)?2.0*TP/(2*TP+FP+FN):0.) << '\n';

    cout << "\n=== AVG RUNTIME (s) ===\n"
         << "Row only        : "<<avg(r_row,N) << '\n'
         << "Column only     : "<<avg(r_col,N) << '\n'
         << "Hybrid optimizer: "<<avg(r_hopt,n_hopt_avail);
    if(n_hopt_avail!=N) cout<<"  (computed on "<<n_hopt_avail<<" / "<<N<<")";
    cout << '\n';

    if(n_fann_avail){
        cout << "FANN model      : "<<avg(r_fann,n_fann_avail)
             <<"  (computed on "<<n_fann_avail<<" / "<<N<<")\n";
    }else{
        cout << "FANN model      : N/A  (no fann_pred in CSV)\n";
    }

    cout << "LightGBM        : "<<avg(r_lgb,N) << '\n'
         << "Optimal (oracle): "<<avg(r_opt,N) << "\n\n";

    LGBM_BoosterFree(booster);
}




/* ─────────── CLI driver ─────────── */
int main(int argc,char*argv[]){
    string base="/home/wuy/query_costs", model="lgb_model.txt";
    vector<string> dirs;
    int trees=400, depth=10; double lr=0.06, subs=0.7, col=0.8;
    string host="127.0.0.1", user="root", pass=""; int port=44444; bool skip=false;

    for(int i=1;i<argc;++i){
        string a(argv[i]);
        if(a.rfind("--data_dirs=",0)==0){ string s=a.substr(12),t; stringstream ss(s); while(getline(ss,t,',')) dirs.push_back(t);}
        else if(a.rfind("--base=",0)==0) base=a.substr(7);
        else if(a.rfind("--trees=",0)==0) trees=stoi(a.substr(8));
        else if(a.rfind("--max_depth=",0)==0) depth=stoi(a.substr(12));
        else if(a.rfind("--lr=",0)==0) lr=stod(a.substr(5));
        else if(a.rfind("--subsample=",0)==0) subs=stod(a.substr(12));
        else if(a.rfind("--colsample=",0)==0) col =stod(a.substr(12));
        else if(a.rfind("--model=",0)==0) model=a.substr(8);
        else if(a=="--skip_train") skip=true;
        else if(a.rfind("--mysql_host=",0)==0) host=a.substr(13);
        else if(a.rfind("--mysql_port=",0)==0) port=stoi(a.substr(13));
        else if(a.rfind("--mysql_user=",0)==0) user=a.substr(13);
        else if(a.rfind("--mysql_pass=",0)==0) pass=a.substr(13);
    }
    if(dirs.empty()){ logE("need --data_dirs=..."); return 1; }

    load_all_index_defs(host,port,user,pass,dirs);
    auto DS=load_dataset(base,dirs);
    if(DS.size()<100){ logE("too few samples"); return 1; }

    train_and_eval(DS,model,trees,depth,lr,subs,col,skip);
    return 0;
}
