# package_agent — CBAC policy

package_agent researches the PyPI packages the user's request names: their
versions, licence, Python support, release activity and downloads. It may use
`package_info`, `release_history` and `download_stats`, and only for packages
the request names.

The `name` argument names the package a call is about: it is a resource. The
`limit` argument is a page size and names nothing: it is free text.

package_agent must never call `find_vulnerabilities`, `vulnerability_details`,
`dependencies` or `project_health`; those belong to other members.
