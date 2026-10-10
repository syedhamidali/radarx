# How to cite

## radarx

Cite radarx with its Zenodo DOI. The DOI covers all versions and always
resolves to the latest release, so citations of every version are counted
together. Add the version you used.

> {{ apa_citation }}

GitHub's "Cite this repository" button gives the same citation in APA and
BibTeX. It is generated from
[`CITATION.cff`](https://github.com/syedhamidali/radarx/blob/main/CITATION.cff).

## The methods you used

Most products of `radarx.retrieve` and `radarx.grid` implement published
methods. Each result records which one, so you can cite the papers behind it.
Three attributes are written on the result (on every sweep of a DataTree):

| Attribute | Content |
| --- | --- |
| `radarx_method` | Short name of the method. It says so where radarx differs from the paper it cites. After several steps, the names are joined with ` \| `, oldest first. |
| `radarx_references` | DOIs of the papers in the docstring's References, separated by a space. A reference without a DOI has a short key such as `doviak-zrnic-1993`. |
| `radarx_version` | The radarx version that wrote the result. |

The attribute `history` gets one line per call with the function name and the
parameters that differ from the defaults. All of them are strings, so they stay in the file when you
write the result to NetCDF.

`radarx.cite` reads the attributes of one result, or of a list of results
from a processing chain, removes duplicates and returns radarx and the papers:

```python
mask = sweep.radarx.echo_mask()
print("\n\n".join(radarx.cite(mask)))
```

Use `style="bibtex"` for BibTeX entries. `radarx.methods(radarx.retrieve.echo_mask)`
or `radarx.methods(mask)` describes in a few lines what a function does and
what it is based on.

Several radarx methods are the package's own constructions that follow the
idea of a paper and not its algorithm. The method name says so. For example,
`echo_mask` is a fuzzy-logic score in the manner of Gourley et al. (2007) and
Krause (2016) with radarx's own memberships and weights. Cite the papers as
the source of the idea and name radarx for the implementation.

## Methods and their references

The list below is generated from the function docstrings and from the
reference registry `radarx/data/references.json`.

```{include} generated/cited_methods.md
```
