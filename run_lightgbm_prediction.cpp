/*****************************************************************************************
 * run_lightgbm_prediction.cpp  – offline predictor (row-vs-column) using a trained
 *                                LightGBM model + the 65-feature extractor you
 *                                just patched (outerRows / eqChainDepth).
 *
 * Build (example):
 *   g++ -O3 -std=c++17 -I$LIGHTGBM/include \
 *       run_lightgbm_prediction.cpp -o run_lgb_predict \
 *       -L$LIGHTGBM/lib -l_lightgbm -lmysqlclient -pthread
 *
 * Run (basic):
 *   ./run_lgb_predict --sql="SELECT …" --model=lgb_model.txt \
 *                     --mysql_db=hybench_sf10 --mysql_tbl=customer
 *
 * Or read the SQL from a file:
 *   ./run_lgb_predict --sql_file=q1.sql …
 *****************************************************************************************/
#include <bits/stdc++.h>
#include <mysql/mysql.h>
#include <LightGBM/c_api.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

/* ───────────────────────────── constants ───────────────────────────── */
constexpr int    NUM_FEATS   = 65;   // 63 + 2 new “row-wins” features
constexpr double COST_THRESH = 5e4;  // purely for an optional fallback print

/* ───────────────────────────── logging ───────────────────────────── */
static void info (const string&s){ cerr<<"[INFO]  "<<s<<'\n'; }
static void warn (const string&s){ cerr<<"[WARN]  "<<s<<'\n'; }
static void die  (const string&s){ cerr<<"[ERR]   "<<s<<'\n'; exit(1); }

/* ───────────────────────────── tiny helpers ───────────────────────── */
static double safe_f(const json& v){
    if(v.is_number())  return v.get<double>();
    if(v.is_string()){ try{ return stod(v.get<string>());}catch(...){ } }
    return 0.0;
}
static double safe_f(const json& o,const char* k){
    return o.contains(k)? safe_f(o[k]) : 0.0;
}
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
    if(v.is_string()){
        string s=v.get<string>(); transform(s.begin(),s.end(),s.begin(),::tolower);
        return s=="yes"||s=="true"||s=="1";
    }
    return false;
}

/* ───────────────────────────── index map (covering tests) ───────────────────────── */
static unordered_map<string, unordered_set<string>> indexCols;

static void load_index_defs_from_db(const string &host,int port,const string &user,
                                    const string &pass,const string &db,
                                    const string &tbl)
{
    MYSQL* c=mysql_init(nullptr);
    if(!mysql_real_connect(c,host.c_str(),user.c_str(),pass.c_str(),
                           db.c_str(),port,nullptr,0))
    { warn("connect "+db+" failed"); mysql_close(c); return; }

    string q="SHOW CREATE TABLE `"+tbl+"`";
    if(mysql_query(c,q.c_str()))
    { warn("SHOW CREATE failed on "+tbl); mysql_close(c); return; }

    MYSQL_RES* r=mysql_store_result(c); MYSQL_ROW row=mysql_fetch_row(r);
    if(!row||!row[1]){ mysql_free_result(r); mysql_close(c); return; }
    string ddl=row[1]; mysql_free_result(r); mysql_close(c);

    static const regex re(R"(KEY\s+`([^`]+)`\s*\(\s*([^)]+)\))",regex::icase);
    smatch m; auto it=ddl.cbegin(), ed=ddl.cend();
    while(regex_search(it,ed,m,re)){
        string idx=m[1], colsRaw=m[2]; unordered_set<string>S;
        string col; stringstream ss(colsRaw);
        while(getline(ss,col,',')){
            col.erase(remove(col.begin(),col.end(),'`'),col.end());
            col.erase(0,col.find_first_not_of(" \t"));
            col.erase(col.find_last_not_of(" \t")+1);
            if(!col.empty()) S.insert(col);
        }
        indexCols[idx]=move(S); it=m.suffix().first;
    }
}

/* ──────────────────────────── feature extractor ──────────────────────────── */
struct Agg{
    double re=0,rp=0,f=0,rc=0,ec=0,pc=0,dr=0,selSum=0,selMin=1e30,selMax=0,
           ratioSum=0,ratioMax=0,maxPrefix=0,minRead=1e30,fanoutMax=0;
    int    cnt=0,cRange=0,cRef=0,cEq=0,cIdx=0,cFull=0,idxUse=0,sumPK=0,
           coverCount=0,maxDepth=0;
    bool   grp=false,ord=false,tmp=false;
    double outerRows=0; int eqChainDepth=0,_curChain=0;
};

static void walk(const json& n, Agg& a, int depth=1){
    if(n.is_object()){
        if(n.contains("table")&&n["table"].is_object()){
            const auto&t=n["table"];
            const auto&ci=t.value("cost_info",json::object());

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
            a.maxPrefix=max(a.maxPrefix,pc);
            a.minRead=min(a.minRead,rc);
            if(re>0){
                double sel=rp/re;
                a.selSum+=sel; a.selMin=min(a.selMin,sel); a.selMax=max(a.selMax,sel);
                a.fanoutMax=max(a.fanoutMax,sel);
            }
            double ratio=ec>0?rc/ec:rc;
            a.ratioSum+=ratio; a.ratioMax=max(a.ratioMax,ratio);

            string at=t.value("access_type","ALL");
            if(at=="range")a.cRange++; else if(at=="ref")a.cRef++;
            else if(at=="eq_ref")a.cEq++; else if(at=="index")a.cIdx++; else a.cFull++;
            if(getBool(t,"using_index")) a.idxUse++;
            if(t.contains("possible_keys")&&t["possible_keys"].is_array())
                a.sumPK+=int(t["possible_keys"].size());

            if(t.contains("used_columns")&&t["used_columns"].is_array()
               && t.contains("key") && t["key"].is_string()){
                string idx=t["key"]; auto it=indexCols.find(idx);
                if(it!=indexCols.end()){
                    bool cover=true;
                    for(auto&u:t["used_columns"])
                        if(!u.is_string()||!it->second.count(u.get<string>()))
                        { cover=false; break; }
                    if(cover) a.coverCount++;
                }
            }

            /* new signals */
            if(a.outerRows==0 && at!="ALL") a.outerRows=re;
            if(at=="eq_ref"){ a._curChain++; a.eqChainDepth=max(a.eqChainDepth,a._curChain);}
            else a._curChain=0;
        }
        if(n.contains("grouping_operation")) a.grp=true;
        if(n.contains("ordering_operation")||getBool(n,"using_filesort")) a.ord=true;
        if(getBool(n,"using_temporary_table")) a.tmp=true;

        for(auto&kv:n.items()) if(kv.key()!="table") walk(kv.value(),a,depth+1);
    }else if(n.is_array()) for(auto&v:n) walk(v,a,depth);

    a.maxDepth=max(a.maxDepth,depth);
}

static bool plan2feat(const json& plan,float f[NUM_FEATS]){
    if(!plan.contains("query_block")) return false;
    const json* qb=&plan["query_block"];
    if(qb->contains("union_result")){
        const auto&s=(*qb)["union_result"]["query_specifications"];
        if(s.is_array()&&!s.empty()) qb=&s[0]["query_block"];
    }
    Agg a; walk(*qb,a);
    if(!a.cnt) return false;

    double inv=1.0/a.cnt;
    int k=0;
    double qCost=safe_f(qb->value("cost_info",json::object()),"query_cost");
    double rootRow=safe_f(*qb,"rows_produced_per_join");

#define PUSH(x) f[k++]=static_cast<float>(x)
    PUSH(log1p_clip(a.re*inv)); PUSH(log1p_clip(a.rp*inv)); PUSH(log1p_clip(a.f*inv));
    PUSH(log1p_clip(a.rc*inv)); PUSH(log1p_clip(a.ec*inv)); PUSH(log1p_clip(a.pc*inv));
    PUSH(log1p_clip(a.dr*inv));

    PUSH(a.cRange*inv); PUSH(a.cRef*inv); PUSH(a.cEq*inv);
    PUSH(a.cIdx*inv);   PUSH(a.cFull*inv); PUSH(a.idxUse*inv);

    PUSH(a.selSum*inv); PUSH(a.selMin); PUSH(a.selMax);
    PUSH(a.maxDepth);   PUSH(a.fanoutMax);

    PUSH(a.grp); PUSH(a.ord); PUSH(a.tmp);

    PUSH(a.ratioSum*inv); PUSH(a.ratioMax);

    PUSH(log1p_clip(qCost)); PUSH(log1p_clip(rootRow));

    PUSH(log1p_clip((a.pc*inv)/max(1e-6,a.rc*inv)));
    PUSH(log1p_clip((a.rc*inv)/max(1e-6,a.re*inv)));
    PUSH(log1p_clip((a.ec*inv)/max(1e-6,a.re*inv)));

    PUSH(a.cnt==1); PUSH(a.cnt>1);
    PUSH(log1p_clip(a.maxDepth*(a.idxUse*inv)));
    PUSH(log1p_clip((a.idxUse*inv)/max(a.cFull*inv,1e-3)));

    PUSH(a.cnt);
    PUSH(a.cnt?double(a.sumPK)/a.cnt:0);
    PUSH(log1p_clip(a.maxPrefix));
    PUSH(log1p_clip(a.minRead<1e30?a.minRead:0));
    PUSH(a.cnt>1?double(a.cnt-1)/a.cnt:0);
    PUSH(rootRow>0?double(a.re)/rootRow:0);
    PUSH(a.selMax-a.selMin);
    PUSH(a.idxUse/double(max(1,a.cRange+a.cRef+a.cEq+a.cIdx)));

    PUSH(qCost);        PUSH(qCost>COST_THRESH);
    PUSH(a.cnt?double(a.coverCount)/a.cnt:0);
    PUSH(a.coverCount==a.cnt);

    PUSH(log1p_clip(a.re*inv)-log1p_clip(a.selSum*inv));
    PUSH(a.cnt); PUSH(log1p_clip(a.cnt));

    PUSH(a.sumPK);
    PUSH(a.cnt?double(a.sumPK)/a.cnt:0);
    PUSH(a.coverCount);
    PUSH(a.cnt?double(a.coverCount)/a.cnt:0);

    PUSH(a.idxUse*inv); PUSH(a.cRange*inv); PUSH(a.cRef*inv);
    PUSH(a.cEq*inv);    PUSH(a.cIdx*inv);   PUSH(a.cFull*inv);

    PUSH(log1p_clip(a.maxPrefix*inv));
    PUSH(log1p_clip(a.minRead<1e30?a.minRead:0));
    PUSH(a.selMax-a.selMin);

    PUSH(a.ratioMax); PUSH(a.fanoutMax);
    PUSH(a.selMin>0?double(a.selMax/a.selMin):0);

    PUSH(log1p_clip(a.outerRows));
    PUSH(a.eqChainDepth);
#undef PUSH
    return k==NUM_FEATS;
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
        { warn("connect "+db+" failed"); mysql_close(c); continue; }

        if(mysql_query(c,"SHOW TABLES"))
        { warn("SHOW TABLES failed in "+db); mysql_close(c); continue; }
        MYSQL_RES *t=mysql_store_result(c);
        MYSQL_ROW r;
        while((r=mysql_fetch_row(t))) load_index_defs_from_db(host,port,user,pass,db,r[0]);
        mysql_free_result(t); mysql_close(c);
    }
}

/* ────────────────────────────────────────────────────────────────────────────
 *  NEW main()  —  robust CSV reader + resilient index-load
 * ────────────────────────────────────────────────────────────────────────── */

struct Rec { std::string db, qid, sql; int use_imci; };

/* --- tiny CSV reader that tolerates quoted commas and embedded new-lines --- */
static bool read_csv_row(std::istream& in, std::vector<std::string>& out)
{
    out.clear(); std::string cell; bool in_q = false;
    for (int ch; (ch = in.get()) != EOF; ) {
        if (ch == '"')                   in_q = !in_q;           /* toggle quote */
        else if (ch == ',' && !in_q)     { out.emplace_back(std::move(cell)); cell.clear(); }
        else if (ch == '\n' && !in_q)    { out.emplace_back(std::move(cell)); return true; }
        else if (ch != '\r')             cell.push_back(char(ch));             /* ignore CR */
    }
    if (!cell.empty() || in_q)           out.emplace_back(std::move(cell));
    return !out.empty();
}
/* --------------- helper: get query_block or diagnose why not ------------- */
static bool extract_query_block(const json& plan, const json*& qb, std::string& why)
{
    if (plan.contains("query_block")) { qb = &plan["query_block"]; return true; }
    if (plan.contains("errmsg"))      { why = plan["errmsg"].get<std::string>(); return false; }
    why = plan.dump(120);             /* first 120 chars of whatever we got */
    return false;
}




/* --------------------------- new main() ---------------------------------- */
int main(int argc,char*argv[])
{

    /* ----- CLI (same flags as before) ----- */
    std::string csv="problem_sqls.csv", model="lgb_model.txt";
    std::string host="127.0.0.1",user="root",pass=""; int port=44444;
    for(int i=1;i<argc;++i){
        std::string a(argv[i]);
        auto eat=[&](const char*s){ size_t L=strlen(s); return a.rfind(s,0)==0? a.substr(L):"";};
        if(auto v=eat("--csv=");!v.empty())   csv=v;
        else if(auto v=eat("--model=");!v.empty()) model=v;
        else if(auto v=eat("--mysql_host=");!v.empty()) host=v;
        else if(auto v=eat("--mysql_port=");!v.empty()) port=stoi(v);
        else if(auto v=eat("--mysql_user=");!v.empty()) user=v;
        else if(auto v=eat("--mysql_pass=");!v.empty()) pass=v;
        else { std::cerr<<"unknown arg "<<a<<'\n'; return 1; }
    }

    /* ----- read CSV (same robust reader as previous message) ----- */
    std::ifstream fin(csv); if(!fin) die("cannot open "+csv);
    std::vector<std::string> row; read_csv_row(fin,row);          /* header */
    struct Rec{std::string db,qid,sql;int use_imci;};
    std::vector<Rec> Q; std::unordered_set<std::string> dbs;
    while(read_csv_row(fin,row) && row.size()>=10){
        Rec r{row[0],row[1],row[9],std::stoi(row[5])};
        if(!r.sql.empty()&&r.sql.front()=='"') r.sql.erase(0,1);
        if(!r.sql.empty()&&r.sql.back()=='"')  r.sql.pop_back();
        if(!r.db.empty()){ Q.push_back(r); dbs.insert(r.db);}      /* drop empty-db rows */
    }
    info("loaded "+std::to_string(Q.size())+" queries from CSV");

    /* ----- index definitions (unchanged) ----- */
    load_all_index_defs(host,port,user,pass,
                        std::vector<std::string>(dbs.begin(),dbs.end()));
    info("index map size = "+std::to_string(indexCols.size()));

    /* ----- LightGBM model ----- */
    BoosterHandle booster=nullptr; int iters=0;
    if(LGBM_BoosterCreateFromModelfile(model.c_str(),&iters,&booster))
        die("cannot load model "+model);
    
    float TAU_STAR = 0.0f;
    
    int correct=0, processed=0;
    for(const auto& r:Q)
    {
        MYSQL* c=mysql_init(nullptr);
        if(!mysql_real_connect(c,host.c_str(),user.c_str(),pass.c_str(),
                               r.db.c_str(),port,nullptr,0))
        { warn("connect "+r.db+" failed"); mysql_close(c); continue; }

        mysql_query(c, "set use_imci_engine=off;");

        /* enlarge packet + relax ONLY_FULL_GROUP_BY */
        mysql_query(c,"SET SESSION max_allowed_packet=67108864");
        mysql_query(c,"SET SESSION sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''))");

        std::string ex="EXPLAIN FORMAT=JSON "+r.sql;
        if(mysql_query(c,ex.c_str()))
        { warn("EXPLAIN fail "+r.qid+": "+mysql_error(c)); mysql_close(c); continue; }

        MYSQL_RES* res=mysql_store_result(c); MYSQL_ROW row_=mysql_fetch_row(res);
        unsigned long* len=mysql_fetch_lengths(res);
        std::string js=(row_&&len)?std::string(row_[0],len[0]):"";
        mysql_free_result(res); mysql_close(c);

        json plan; try{ plan=json::parse(js);}catch(...){
            warn("bad JSON for "+r.qid); continue; }

        const json* qb=nullptr; std::string why;
        if(!extract_query_block(plan,qb,why)){
            warn("plan missing query_block ("+r.qid+") "+why.substr(0,80));
            /*  fallback = cost threshold  */
            bool act_col = r.use_imci==1;
            bool pred_col = plan.contains("cost_info") &&
                            safe_f(plan["cost_info"],"query_cost") > COST_THRESH;
            if(pred_col==act_col) ++correct;
            ++processed;
            std::cout<<r.db<<':'<<r.qid<<"  pred="<<(pred_col?"COLUMN":"ROW")
                     <<"*  actual="<<(act_col?"COLUMN":"ROW")<<"  (fallback)\n";
            continue;
        }

        float feat[NUM_FEATS]={};
        if(!plan2feat(plan,feat)){ warn("unexpected extract failure "+r.qid); continue; }

        double prob=0; int64_t out_len=0;
        LGBM_BoosterPredictForMat(booster,feat,C_API_DTYPE_FLOAT32,
                                  1,NUM_FEATS,1,C_API_PREDICT_NORMAL,
                                  -1,0,"",&out_len,&prob);

        
        bool pred_col= prob>TAU_STAR, act_col=r.use_imci==1;

        if(pred_col==act_col) ++correct;
        ++processed;

        std::cout<<r.db<<':'<<r.qid<<"  pred="<<(pred_col?"COLUMN":"ROW")
                 <<"  actual="<<(act_col?"COLUMN":"ROW")
                 <<"  prob="<<std::fixed<<std::setprecision(3)<<prob<<'\n';
    }

    std::cout<<"\nProcessed "<<processed<<" / "<<Q.size()
             <<"   Accuracy "<<correct<<'/'<<processed<<" ("
             <<std::fixed<<std::setprecision(2)
             <<(processed?100.0*correct/processed:0.0)<<"%)\n";
    LGBM_BoosterFree(booster);
    return 0;
}
