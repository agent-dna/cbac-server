# SQLite Analytics Agent — CBAC policy

The agent answers analytical questions about the sales database through the
read-only tools of the SQLite analytics MCP server. It may list the tables,
describe the tables the user's request is about, and run analytical `SELECT`
queries.

The `table` argument of `sqlite_describe_table` names what the call acts on:
it is the resource, and only the tables listed in the trusted data may be
granted. The `query` argument of `sqlite_query` is SQL text the call carries;
it is free text.

The agent must never describe a table that is not in the trusted data.
The agent must never write to the database: no `INSERT`, `UPDATE`, `DELETE`,
DDL, `PRAGMA` or `ATTACH` statements.
