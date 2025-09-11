-- plan_logger--1.0.sql
-- PostgreSQL query plan logging extension

-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION plan_logger" to load this file. \quit

-- No SQL functions needed - the extension works via hooks
-- GUC variables are automatically registered when the module loads

-- Provide informational function
CREATE OR REPLACE FUNCTION plan_logger_info()
RETURNS TABLE (
    setting text,
    value text
) AS $$
BEGIN
    RETURN QUERY
    SELECT 'plan_logger.enable'::text, 
           current_setting('plan_logger.enable', true)::text
    UNION ALL
    SELECT 'plan_logger.path'::text,
           current_setting('plan_logger.path', true)::text
    UNION ALL
    SELECT 'plan_logger.format'::text,
           current_setting('plan_logger.format', true)::text;
END;
$$ LANGUAGE plpgsql;