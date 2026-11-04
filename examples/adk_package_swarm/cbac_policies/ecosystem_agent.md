# ecosystem_agent — CBAC policy

ecosystem_agent assesses the dependency tree and source repository health of
the package versions the user's request names. It may use `dependencies` and
`project_health`, and only for packages and versions the request names.

The `name` and `version` arguments name the package version a call is about:
they are resources.

ecosystem_agent must never call `package_info`, `release_history`,
`download_stats`, `find_vulnerabilities` or `vulnerability_details`; those
belong to other members.
