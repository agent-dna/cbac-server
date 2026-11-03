-- Automatically run on first database creation by docker-entrypoint-initdb.d.
-- Enables the required Postgres extensions for the CBAC service, and creates
-- the database OpenFGA keeps its tuples in (migrated by the openfga-migrate
-- compose service).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;

CREATE DATABASE openfga;
