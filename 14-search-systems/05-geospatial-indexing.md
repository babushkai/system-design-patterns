# Geospatial Indexing

## TL;DR

A spatial index answers "what is near here" or "what contains this point" without comparing the query against every object. The difficulty is that ordinary indexes sort by one dimension and location has two: no ordering of points on a line keeps all nearby points adjacent. Every practical design is therefore a way to map two dimensions onto structures that existing storage already handles, and every one of them leaks at the edges.

Two families cover almost all production use. **Cell-based schemes** (geohash, S2, H3) cut the surface into a hierarchy of cells with integer or string identifiers, so a location becomes a key and a region becomes a set of key ranges that any ordered store or hash map can serve. **Tree-based schemes** (R-trees, quadtrees) index bounding rectangles directly and suit extended shapes such as polygons and routes. Both are used the same way: a coarse **filter** step that returns candidates cheaply, followed by an exact **refine** step. The design work is choosing the cell level or tree fan-out so the filter is neither too loose nor too many lookups, handling density skew between a city center and a desert, and deciding how fast-moving objects are written. [Uber](../08-case-studies/03-uber.md) and [Google Maps](../08-case-studies/09-google-maps.md) show these structures inside complete systems.

## Primary Evidence and Scope

| Source | What it establishes | Boundary |
|---|---|---|
| Finkel and Bentley (1974) | Quadtrees: recursive four-way subdivision for composite keys | The original point structure; many variants followed |
| Guttman (SIGMOD 1984) | R-trees: a balanced tree of minimum bounding rectangles with dynamic insert and delete | Node overlap grows with poor splits; later variants improve packing |
| Samet (ACM Computing Surveys, 1984) | A taxonomy of quadtree-style hierarchical structures and their operations | Survey, not a performance claim |
| S2 Geometry documentation | A cell hierarchy on the sphere using a cube projection and a Hilbert curve, with 64-bit cell identifiers and 31 levels | One library's design |
| H3 documentation | A hexagonal hierarchical grid with 16 resolutions, built on an icosahedron | Hexagons do not nest exactly across resolutions |
| Redis and PostGIS documentation | Geohash-in-sorted-set and GiST R-tree indexing as deployed in two common stores | Product behavior for the documented versions |

Sizes and rates in the worked sections are illustrative.

## Queries and Contract

Spatial workloads reduce to a few query shapes with different index needs:

| Query | Example | What the index must do |
|---|---|---|
| Radius or k-nearest | Drivers within 2 km; five closest stores | Enumerate cells around a point, expand until enough results |
| Bounding box | Everything in the visible map tile | Range over cells or rectangles intersecting a box |
| Point in polygon | Which delivery zone or tax region contains this address | Find candidate polygons, then run an exact containment test |
| Spatial join | Count trips per neighborhood | Assign each point to a region at scale |

The contract to write down: coordinate reference and axis order, distance definition (straight-line on the sphere, or travel distance on a network), maximum staleness for moving objects, result exactness (may a result just outside the radius appear?), and the largest region one query may cover.

Straight-line distance is rarely the product requirement. The nearest driver by great-circle distance may be across a river with no bridge. A spatial index produces candidates; ranking by travel time belongs to a routing engine.

## The Filter and Refine Pattern

```mermaid
flowchart LR
    Q[Query region] --> COV[Cover with cells<br/>or descend the tree]
    COV --> CAND[Candidate set<br/>superset of the answer]
    CAND --> EXACT[Exact geometry test<br/>distance or containment]
    EXACT --> R[Results]
```

No cell grid matches a circle or an arbitrary polygon, so the index returns a superset and exact geometry removes the excess. The cost is `lookups + candidates x exact_test_cost`. Coarse cells mean few lookups and many false candidates; fine cells mean the reverse. The best level is the one where the cells are roughly the size of the query region, which is why systems index the same object at several levels or choose the level per query.

## Geohash: Interleaved Bits as a String

Geohash divides the longitude range in half, then the latitude range, and repeats, emitting one bit per split. Interleaving the two bit streams traces a Z-order curve; encoding five bits per character in base 32 yields a string. Truncating the string yields the enclosing cell, so prefix matching is containment.

| Length | Approximate cell size at the equator |
|---|---|
| 5 | 4.9 km x 4.9 km |
| 6 | 1.2 km x 0.61 km |
| 7 | 153 m x 153 m |
| 8 | 38 m x 19 m |

Its appeal is that it needs nothing special from the store: a string column with an ordinary index serves prefix queries. **Documented, Redis:** the geospatial commands store each member in a sorted set with a geohash-derived integer as the score, and radius queries become score-range scans followed by distance filtering.

Its limits define when to move on:

- **Boundary discontinuities.** Two points a meter apart on opposite sides of a cell boundary can share no prefix at all. A correct proximity query always searches the target cell and its eight neighbors.
- **Shape and area distortion.** Cells are rectangles in latitude and longitude, so their ground width shrinks toward the poles.
- **Z-order jumps.** The curve has long jumps between quadrants, so a small region can map to many disjoint key ranges.

## S2: Cells on a Sphere With a Hilbert Curve

**Documented:** S2 projects the sphere onto the six faces of a cube, subdivides each face as a quadtree to 30 levels below the face, and numbers cells along a Hilbert curve. A cell identifier is a 64-bit integer, and a parent's identifier range contains all of its descendants. Leaf cells are on the order of a square centimeter.

Three properties matter for system design:

1. **Containment is a range check.** All descendants of a cell fall in one contiguous integer interval, so "everything in this cell" is a single range scan on a sorted key.
2. **Better locality.** The Hilbert curve has no long jumps, so a region covers with fewer, longer ranges than a Z-order curve would need.
3. **Region covering is a library function.** A coverer returns a small set of cells, at mixed levels, that covers an arbitrary region within a requested cell budget. The budget is the filter-versus-lookups dial made explicit.

Cells at one level have similar but not equal area across the sphere, without the polar collapse of latitude and longitude grids.

## H3: Hexagons for Movement and Aggregation

**Documented:** H3 tiles the globe with hexagons at 16 resolutions, plus 12 pentagons per resolution required by the underlying icosahedron. Cell identifiers are 64-bit integers.

Hexagons are chosen for one geometric reason: every neighbor shares an edge and lies at the same center distance. A square cell has four edge neighbors and four corner neighbors at different distances, which biases anything that spreads between cells. For demand heat maps, surge areas, smoothing, and flow between adjacent areas, a single neighbor class removes that bias.

The cost is hierarchy. Seven child hexagons approximate their parent but do not tile it exactly, so parent and child aggregates differ slightly at the edges. Use H3 where the cell is the unit of analysis and a prefix-range store is not the access path. Use S2 where exact nesting and range scans are.

## R-Trees: Indexing Extents

Cells suit points. A river, a delivery zone, or a building footprint has extent, and assigning it to cells means storing it in every cell it touches. An R-tree indexes extents directly.

Each leaf entry holds an object's minimum bounding rectangle. Each internal node holds the bounding rectangle of its children. A search descends every child whose rectangle intersects the query. Because sibling rectangles may overlap, a search can follow several paths, and the quality of the tree is the amount of overlap its insertion and split heuristics leave behind. Bulk-loading a sorted dataset packs far better than inserting row by row.

**Documented, PostGIS:** spatial indexes are R-trees implemented on PostgreSQL's GiST framework, and an indexed query first matches bounding boxes, then applies the exact geometry predicate. That is the filter and refine pattern inside one engine.

R-trees fit relational stores holding polygons and mixed geometry with moderate write rates. They are a poor fit for millions of position updates per minute, where every update rebalances a shared tree.

## Moving Objects and Write-Heavy Location State

A fleet of vehicles reporting position every few seconds is a different problem from indexing addresses. The index is rewritten continuously and almost every write supersedes a previous one.

**Reference design:**

- Keep the live index in memory, keyed by cell, with each object's last position and timestamp. Durable history is a separate append-only stream.
- Update in place by removing the object from its old cell and adding it to the new one. Most updates do not change cell at a sensible level, so the common case is a single hash-map write.
- Expire entries by timestamp so an object that stops reporting disappears on its own. The reasoning matches [Presence as Derived Multi-Session State](../07-real-time/06-presence.md): absence of updates is the signal.
- Shard by a coarse geographic unit (city or a low-level cell), not by object identifier, so a proximity query touches one shard. Queries near shard borders must consult the neighbors.

Density skew is the standing hazard. A fixed cell level that puts ten objects per cell downtown puts zero in most of the map and ten thousand at a stadium on event night. Either choose the level per region from observed density, or make the search adaptive: start at the target cell, expand ring by ring, and stop when `k` results are found or a maximum radius is reached. See [Partitioning Strategies](../02-distributed-databases/05-partitioning-strategies.md) for the general treatment of hot partitions.

## Capacity Model

All inputs are illustrative.

**Live fleet state.** One million active vehicles report every 4 seconds.

```text
writes/s        = 1,000,000 / 4           = 250,000
live state      = 1,000,000 x 64 bytes    = 64 MB
```

The data is tiny and the write rate is not. This workload is sized by update throughput and fan-out to shards, never by storage.

**Cell level for a 2 km radius query.** The circle covers 12.6 square kilometers.

```text
geohash length 5   4.9 km cells   up to 4 cells     candidate area up to 96 km^2   about 8x the circle
geohash length 6   1.2 x 0.61 km  about 28 cells    candidate area about 21 km^2   about 1.7x the circle
geohash length 7   153 m cells    about 700 cells   candidate area close to exact  lookups dominate
```

Length 6 is the reasonable middle: tens of lookups and under twice the necessary candidates. Length 5 reads eight times the needed objects in a dense area; length 7 turns one query into hundreds of key lookups. The right answer changes with the radius, which is the argument for a coverer that mixes levels.

## Failure Traces

### The boundary miss

A store locator hashes the user's position to a six-character geohash and returns stores sharing that prefix. A user standing 50 meters from a store on the other side of a cell boundary is told the nearest store is 3 kilometers away. The prefix lookup was treated as the answer instead of as a filter. Always search neighbors, then rank by exact distance.

### Longitude and latitude swapped

One service emits `(lat, lng)` and another expects `(lng, lat)`, as several geometry formats specify. Points land in the ocean or in another hemisphere, and the index returns empty results with no error. Validate ranges at the boundary (latitude is limited to plus or minus 90), name the fields, and include a known landmark in integration tests.

### The stadium cell

A concert ends and 40,000 riders open the app inside one cell. Every request scans the same cell's driver list, and the shard holding it saturates while its neighbors idle. Cap candidates per cell, use finer cells where density is high, and shed or queue requests per cell before the shard fails. The admission mechanics are in [Rate Limiting](../06-scaling/05-rate-limiting.md).

### Stale positions match

A vehicle loses connectivity in a tunnel. Its last position remains in the index and it is offered trips for ten minutes. Enforce a maximum age in the query path, not only in a background sweeper, so correctness does not depend on cleanup timing.

### Antimeridian and pole queries

A bounding box crossing longitude 180 is expressed with a minimum longitude greater than its maximum, and naive range logic returns the complement. Split such boxes into two, and test polar and antimeridian regions explicitly even when the product does not operate there, because test data and bad inputs do.

## Operations and Verification

Measure candidates returned per result kept (filter precision), cells or nodes visited per query, per-cell object counts at the high percentiles, update lag for moving objects, and the age distribution of matched positions.

Verify against brute force: generate random points and regions, compute the exact answer by scanning everything, and compare. Include points on cell boundaries, regions spanning shard borders, the antimeridian, and high-latitude areas. For moving objects, replay a recorded trace at several times real speed and check that no query returns an object older than the staleness bound.

## Decision Framework

| Situation | Choice | Reason |
|---|---|---|
| Proximity on points in a key-value or sorted store | S2 cell identifiers with a region coverer | Containment is one range scan; Hilbert order needs few ranges |
| Proximity with minimal dependencies, moderate accuracy | Geohash with neighbor search | Works on any string index; mind boundaries and latitude distortion |
| Aggregation, heat maps, surge areas, flows between adjacent areas | H3 | One neighbor class at uniform distance |
| Polygons, routes, mixed geometry in a relational database | R-tree through the database's spatial index | Indexes extents without duplicating objects across cells |
| Millions of position updates per minute | In-memory cell map with expiry, sharded by region | Updates are single-cell writes; no shared tree to rebalance |
| Density varies by orders of magnitude | Adaptive level or ring expansion | A fixed level is wrong almost everywhere |
| "Nearest" means travel time | Spatial index for candidates, routing engine to rank | Straight-line distance ignores the network |

Then confirm four things: the index is used only as a filter, the neighbor and boundary cases are covered, staleness is enforced on read, and the largest query region is bounded.

## References

- [Finkel and Bentley, *Quad Trees: A Data Structure for Retrieval on Composite Keys* (Acta Informatica, 1974)](https://doi.org/10.1007/BF00288933)
- [Guttman, *R-Trees: A Dynamic Index Structure for Spatial Searching* (SIGMOD 1984)](https://doi.org/10.1145/602259.602266)
- [Samet, *The Quadtree and Related Hierarchical Data Structures* (ACM Computing Surveys, 1984)](https://doi.org/10.1145/356924.356930)
- [S2 Geometry, *S2 Cell Hierarchy*](https://s2geometry.io/devguide/s2cell_hierarchy)
- [H3 documentation](https://h3geo.org/)
- [Uber Engineering, *H3: Uber's Hexagonal Hierarchical Spatial Index*](https://www.uber.com/blog/h3/)
- [Redis documentation, *Geospatial*](https://redis.io/docs/latest/develop/data-types/geospatial/)
- [PostGIS workshop, *Spatial Indexing*](https://postgis.net/workshops/postgis-intro/indexing.html)

---

## Related Chapters

- [Uber](../08-case-studies/03-uber.md): cell-indexed supply and demand inside a complete dispatch system
- [Google Maps](../08-case-studies/09-google-maps.md): tiles, places, and routing, where straight-line candidates meet the road graph
- [Partitioning Strategies](../02-distributed-databases/05-partitioning-strategies.md): hot cells are hot partitions, with the same mitigations
- [Vector Retrieval Systems](./03-vector-search.md): approximate nearest neighbors in hundreds of dimensions, where grids stop working
- [Presence as Derived Multi-Session State](../07-real-time/06-presence.md): expiry by silence, the same rule that removes a vehicle from the map
