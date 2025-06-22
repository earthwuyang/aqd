g++ -O3 -std=c++11 -I/usr/local/include \
    lightgbm_row_column.cpp -o lightgbm_row_column \
    -I/home/wuy/software/json-develop/single_include/nlohmann -L/usr/local/lib -L/home/wuy/software/LightGBM/ -I/home/wuy/software/LightGBM/include -l_lightgbm  -I/usr/include/mysql -L/usr/lib64/mysql -lmysqlclient -lstdc++fs -pthread
