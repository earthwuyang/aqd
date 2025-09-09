-- Enable AQD feature logging
SET aqd.enable_feature_logging = true;
SET aqd.log_format = 1;
SET aqd.feature_log_path = '/tmp/aqd_features_test.json';

-- Show current settings
SHOW aqd.enable_feature_logging;
SHOW aqd.log_format;
SHOW aqd.feature_log_path;

-- Run a simple query
SELECT * FROM tpch_sf1.nation LIMIT 5;