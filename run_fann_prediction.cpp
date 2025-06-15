#include <mysql/mysql.h>
#include <fann.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

double convert_data_size_to_numeric(std::string s){
    if(s.empty()) return 0.0;
    s.erase(std::find_if(s.rbegin(),s.rend(),
         [](unsigned char c){return !std::isspace(c);} ).base(),s.end());
    char suff=s.back(); double f=1.0;
    if(suff=='G'){f=1e9; s.pop_back();}
    else if(suff=='M'){f=1e6; s.pop_back();}
    else if(suff=='K'){f=1e3; s.pop_back();}
    try{ return std::stod(s)*f; }catch(...){ return 0.0;}
}

inline double safeLog1p(double v) {
    return std::log1p(std::max(0.0, v));
}

double parseNumber(const json &j, const std::string &key) {
    if (!j.contains(key)) return 0.0;
    try {
        if (j[key].is_string()) return std::stod(j[key].get<std::string>());
        if (j[key].is_number()) return j[key].get<double>();
    } catch (...) {}
    return 0.0;
}
bool extractFeaturesFromPlan(const json &j, fann_type input[8]) {
    if (!j.contains("query_block")) return false;
    const json &qb = j["query_block"];

    // 累加器
    double sumRe=0, sumRp=0, sumF=0;
    double sumRc=0, sumEc=0, sumPc=0, sumDr=0;
    int count=0;

    // 递归 lambda：遇到 object 检查有没有 "table"，然后继续向下
    std::function<void(const json&)> rec = [&](const json &node) {
        if (node.is_object()) {
            // 如果包含 table，就提取特征
            if (node.contains("table") && node["table"].is_object()) {
                const json &tbl = node["table"];
                const json &ci  = tbl.value("cost_info", json::object());

                double re = parseNumber(tbl, "rows_examined_per_scan");
                double rp = parseNumber(tbl, "rows_produced_per_join");
                double f  = parseNumber(tbl, "filtered");
                double rc = parseNumber(ci,  "read_cost");
                double ec = parseNumber(ci,  "eval_cost");
                double pc = parseNumber(ci,  "prefix_cost");

                // data_read_per_join 可能带 G/M/K
                double dr = 0.0;
                if (ci.contains("data_read_per_join")) {
                    if (ci["data_read_per_join"].is_string())
                        dr = convert_data_size_to_numeric(
                                ci["data_read_per_join"].get<std::string>());
                    else
                        dr = parseNumber(ci, "data_read_per_join");
                }

                sumRe += re; sumRp += rp; sumF  += f;
                sumRc += rc; sumEc += ec; sumPc += pc;
                sumDr += dr;
                ++count;
            }
            // 遍历所有子字段
            for (auto &kv : node.items())
                rec(kv.value());
        }
        else if (node.is_array()) {
            for (auto &elem : node) rec(elem);
        }
    };

    // 启动递归
    rec(qb);

    if (count == 0) return false;  // 找不到任何 table

    // 计算平均并做 log1p
    auto avg_log1p = [&](double s){
        return safeLog1p(s / double(count));
    };

    input[0] = static_cast<fann_type>(avg_log1p(sumRe));
    input[1] = static_cast<fann_type>(avg_log1p(sumRp));
    input[2] = static_cast<fann_type>(avg_log1p(sumF));
    input[3] = static_cast<fann_type>(avg_log1p(sumRc));
    input[4] = static_cast<fann_type>(avg_log1p(sumEc));
    input[5] = static_cast<fann_type>(avg_log1p(sumPc));
    input[6] = static_cast<fann_type>(avg_log1p(sumDr));

    // 最后一个特征是整个 query 的 query_cost
    double qc = 0.0;
    if (qb.contains("cost_info"))
        qc = parseNumber(qb["cost_info"], "query_cost");
    input[7] = static_cast<fann_type>(safeLog1p(qc));

    // Debug 输出
    std::cout << "[Feature Debug] averaged log1p features over "
              << count << " tables:\n"
              << "  rows_examined_per_scan = " << input[0] << "\n"
              << "  rows_produced_per_join = " << input[1] << "\n"
              << "  filtered               = " << input[2] << "\n"
              << "  read_cost              = " << input[3] << "\n"
              << "  eval_cost              = " << input[4] << "\n"
              << "  prefix_cost            = " << input[5] << "\n"
              << "  data_read_per_join     = " << input[6] << "\n"
              << "  query_cost             = " << input[7] << "\n";

    return true;
}



int main() {
    const char *host = "127.0.0.1";
    const char *user = "root";
    const char *pass = "";
    const char *db   = "hybench_sf1";
    unsigned int port = 44444;

    MYSQL *conn = mysql_init(nullptr);
    if (!mysql_real_connect(conn, host, user, pass, db, port, nullptr, 0)) {
        std::cerr << "MySQL connection failed: " << mysql_error(conn) << std::endl;
        return 1;
    }

    const char *set_query = "set use_imci_engine=off";
    if (mysql_query(conn, set_query) != 0) {
        std::cerr << "Query failed: " << mysql_error(conn) <<std::endl;
        mysql_close(conn);
        return 1;
    }
    // const char *query = "EXPLAIN FORMAT=JSON SELECT COUNT(*) FROM transfer";
    // const char *query = "explain format=json SELECT AVG(`savingAccount`.`balance`) as agg_0 FROM `loantrans` JOIN `loanapps` ON `loantrans`.`appID` = `loanapps`.`id` JOIN `customer` ON `loantrans`.`applicantID` = `customer`.`custID` JOIN `checking` ON `customer`.`custID` = `checking`.`sourceID` JOIN `company` ON `customer`.`companyID` = `company`.`companyID` JOIN `savingAccount` ON `customer`.`custID` = `savingAccount`.`userID`";
    const char *query = "explain format=json SELECT `customer`.`name`, SUM(`checking`.`targetID`) as agg_0, AVG(`company`.`Isblocked`) as agg_1, COUNT(*) as agg_2 FROM `company` JOIN `customer` ON `company`.`companyID` = `customer`.`companyID` JOIN `loanapps` ON `customer`.`custID` = `loanapps`.`applicantID` JOIN `transfer` ON `customer`.`custID` = `transfer`.`targetID` JOIN `checking` ON `customer`.`custID` = `checking`.`sourceID`  GROUP BY `customer`.`name` ORDER BY `customer`.`name` LIMIT 100;";
    if (mysql_query(conn, query) != 0) {
        std::cerr << "Query failed: " << mysql_error(conn) << std::endl;
        mysql_close(conn);
        return 1;
    }

    MYSQL_RES *res = mysql_store_result(conn);
    if (!res) {
        std::cerr << "Failed to get result: " << mysql_error(conn) << std::endl;
        mysql_close(conn);
        return 1;
    }

    MYSQL_ROW row = mysql_fetch_row(res);
    if (!row || !row[0]) {
        std::cerr << "No result row" << std::endl;
        mysql_free_result(res);
        mysql_close(conn);
        return 1;
    }

    unsigned long *lengths = mysql_fetch_lengths(res);
    std::string json_text(row[0], lengths[0]);

    std::cout << "[DEBUG] Raw EXPLAIN JSON:\n"
                << json_text << "\n\n";



    json plan;
    try {
        plan = json::parse(row[0]);
    } catch (...) {
        std::cerr << "Failed to parse JSON explain plan" << std::endl;
        mysql_free_result(res);
        mysql_close(conn);
        return 1;
    }

    fann_type input[8];
    if (!extractFeaturesFromPlan(plan, input)) {
        std::cerr << "Failed to extract features from plan" << std::endl;
        mysql_free_result(res);
        mysql_close(conn);
        return 1;
    }

    const char *model_path = "checkpoints/best_mlp_no_tg.net";
    struct fann *ann = fann_create_from_file(model_path);
    if (!ann) {
        std::cerr << "Failed to load FANN model from: " << model_path << std::endl;
        return 1;
    }

    fann_type *output = fann_run(ann, input);
    if (output) {
        std::cout << "FANN prediction output: " << output[0]
                  << " => Use " << (output[0] >= 0.5f ? "Column Store" : "Row Store") << std::endl;
    } else {
        std::cerr << "FANN prediction failed" << std::endl;
    }

    fann_destroy(ann);
    mysql_free_result(res);
    mysql_close(conn);
    return 0;
}
