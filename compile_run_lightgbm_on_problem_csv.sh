g++ -O3 -std=c++11 -I/usr/local/include \
    run_lightgbm_on_problem_csv.cpp -o run_lightgbm_on_problem_csv \
    -I/home/wuy/software/json-develop/single_include/nlohmann -L/usr/local/lib -L/home/wuy/software/LightGBM/ -I/home/wuy/software/LightGBM/include -l_lightgbm  -I/usr/include/mysql -L/usr/lib64/mysql -lmysqlclient -lstdc++fs -pthread
