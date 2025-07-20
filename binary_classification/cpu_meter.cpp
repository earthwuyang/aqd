/* cpu_meter.cpp */
#include "cpu_meter.h"
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <chrono>
#include <cstring>
#include <cstdio>

namespace {
inline unsigned long long read_jiffies(const std::string& path,int col)
{
    int fd = ::open(path.c_str(), O_RDONLY|O_CLOEXEC);
    if(fd<0) return 0;
    char buf[512]={0}; ::read(fd,buf,sizeof(buf)-1); ::close(fd);
    int c=1; unsigned long long v=0; char* p=buf;
    while(c<=col && *p){
        if(*p==' '||*p=='\n'){ ++c; ++p; continue; }
        if(c==col){ v=strtoull(p,&p,10);}
        else while(*p&&*p!=' '&&*p!='\n') ++p;
    }
    return v;
}
}

CpuMeter& CpuMeter::instance()
{
    static CpuMeter inst;
    return inst;
}

CpuMeter::CpuMeter()
 : pid_(::getpid()),
   hz_(::sysconf(_SC_CLK_TCK)),
   ncpu_(::sysconf(_SC_NPROCESSORS_ONLN)),
   last_({0.0,0.0}),
   bg_(&CpuMeter::loop,this)
{}

CpuMeter::~CpuMeter(){
    stop_ = true;
    if(bg_.joinable()) bg_.join();
}

std::pair<double,double> CpuMeter::read_once() const
{
    /* 1) 枚举线程 ID */
    std::vector<pid_t> col,row;
    std::string dir = "/proc/"+std::to_string(pid_)+"/task";
    if(DIR* d=::opendir(dir.c_str())){
        dirent* e;
        while((e=::readdir(d))){
            if(e->d_type!=DT_DIR) continue;
            pid_t tid = atoi(e->d_name);
            if(tid<=0) continue;
            std::string comm = dir + "/" + e->d_name + "/comm";
            char name[64]={0};
            int fd=::open(comm.c_str(),O_RDONLY|O_CLOEXEC);
            if(fd>=0){
                ::read(fd,name,sizeof(name)-1); ::close(fd);
                if(strstr(name,"imci[")) col.push_back(tid);
                else                      row.push_back(tid);
            }
        }
        ::closedir(d);
    }
    auto sum =[&](const std::vector<pid_t>& v){
        unsigned long long tot=0;
        for(pid_t t:v){
            std::string st="/proc/"+std::to_string(pid_)+"/task/"+
                            std::to_string(t)+"/stat";
            tot+=read_jiffies(st,14)+read_jiffies(st,15);
        }
        return tot;
    };
    unsigned long long row_j = sum(row);
    unsigned long long col_j = sum(col);

    static unsigned long long last_row=0,last_col=0;
    unsigned long long d_row=row_j-last_row, d_col=col_j-last_col;
    last_row=row_j; last_col=col_j;
    double denom = hz_*ncpu_;
    return {100.0*d_row/denom, 100.0*d_col/denom};
}

void CpuMeter::loop()
{
    using namespace std::chrono_literals;
    while(!stop_){
        last_ = read_once();
        std::this_thread::sleep_for(1s);
    }
}
