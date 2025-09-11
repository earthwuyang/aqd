# plan_logger Makefile

MODULE_big = plan_logger
OBJS = plan_logger.o

EXTENSION = plan_logger
DATA = plan_logger--1.0.sql
PGFILEDESC = "plan_logger - PostgreSQL query plan logging extension"

# Use pg_config from our PostgreSQL installation
PG_CONFIG = /home/wuy/DB/pg_duckdb_postgres/pgsql/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)