# Design Doc

_author_: Alex Merose

_co-author_: Agent TBD. (agent specifics goes here).

_created_: 2026-07-19


## Goals & High Level Design

Goal: I want to create a generic component for XQL-style autograd that can be integrated into composable database systems
(like DataFusion, DuckDB, maybe Postgres, etc.). 

- I created a prototype for [JAX-style symbolic automatic differentiation](https://github.com/jax-ml/jax/blob/main/jax/interpreters/ad.py) within [Xarray-SQL](https://github.com/xqlsystems/xarray-sql): https://github.com/xqlsystems/xarray-sql/pull/192. This whole design doc is based on this prototype, so before proceeding, please review it.
- I'd like to use the [substrait.io](https://substrait.io/) query engine protocol and design system to create portable, generic database components to bring this autograd implementation to a wide audience of databases.
  - Like the prototype, I'd like to expose a few core, row-level UDFs that can be installed in each database that uses this extension, namely `grad`, `jvp`, and `vjp`. 
  - I'd like to follow substrait's recommendations for creating UDFs:
    - https://substrait.io/expressions/user_defined_functions/ 
    - https://substrait.io/extensions/
  - I'd like to implement this extension in Rust, possibly using substrait-rust: https://github.com/substrait-io/substrait-rs
- In this package, I think I'd like to research possible deployment systems into common DB targets -- let's say the first two targets are DataFusion and DuckDB, with Postgres coming later. I think the best way to do this would be to use a UDF registration system.
  - DuckDB Python UDF API: https://duckdb.org/docs/lts/clients/python/function
  - DataFusion-Python UDF API: https://datafusion.apache.org/python/user-guide/common-operations/udf-and-udfa.html
  - DataFusion Rust UDF Guide: https://datafusion.apache.org/library-user-guide/functions/adding-udfs.html
  - DuckDB Community Extensions Guide: https://duckdb.org/community_extensions/; rust extension template: https://github.com/duckdb/extension-template-rs/
  - Postgres Extension in Rust: https://github.com/pgcentralfoundation/pgrx (this can come later on when we support arrays in the XQL pattern within Postgres).
- It's possible that this repo should be the core implementation as a library and we have lightweight wrappers for the low level extensions (all written in Rust, maybe somtimes with Python bindings) to target other RDBMS like the above. To simplify development, I think this could be a monorepo pattern where we publish a core crate with the autograd impl and then integration crates for each DB system.
- To verify success of these extensions, this project should seamlessly integrate with [Xarray-SQL](https://github.com/xqlsystems/xarray-sql) (a DataFusion Python project) and [DuckDB-Zarr](https://github.com/xqlsystems/duckdb-zarr) (A Rust-based community extension). These aren't toy integrations, they're active projects with real users. 
- If we surface public extensions and have to choose a name, I prefer the term "ddx" or "ddx db". e.g. `pip install ddxdb` or `INSTALL ddx FROM community;`. This is an homage to my thoughts about how machine learning models are differential databases.