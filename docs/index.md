---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for radarx, with links to the rest
      of the site.
html_theme.sidebar_secondary.remove: true
---

**Release:** {{release}}\
**Date:** {{today}}

```{include} ../README.md
```

```{toctree}
:maxdepth: 3
:caption: Contents

installation
usage
modules
contributing
authors
history
reference
```

Indices and tables
==================
- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
