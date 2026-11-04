# security_agent — CBAC policy

security_agent looks up known vulnerabilities in the package versions the
user's request names, and the versions that fix them. It may use
`find_vulnerabilities` and `vulnerability_details`, and only for packages and
versions the request names.

The `name` and `version` arguments name the package version a call is about:
they are resources. The `vuln_id` argument is an identifier the server
returned in an earlier result and names nothing the user chose: it is free
text.

security_agent must never call `package_info`, `release_history`,
`download_stats`, `dependencies` or `project_health`; those belong to other
members.
