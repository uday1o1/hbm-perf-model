# Catalog schema

`schema_version`: 1

All catalogs are UTF-8 CSV files with a header row, one row per entity, a stable lowercase `id`, an HTTPS `source_url` (several URLs are separated by ` ; `), and a free-text `notes` column.
`hbmperf catalog validate` enforces the rules below; `src/hbmperf/catalog.py` holds the exact column lists.

## Units

Every numeric column name carries its unit, and no column accepts another unit.

| Suffix | Unit |
| --- | --- |
| `_gb` | decimal gigabytes (1e9 bytes) |
| `_gb_per_s` | decimal gigabytes per second |
| `_gbit_per_s`, `_mbit_per_s` | gigabits or megabits per second (per pin for memory interfaces) |
| `_tflops`, `_tops` | 1e12 dense operations per second (no structured sparsity) |
| `_ck` | DRAM command-clock cycles |
| `_ps`, `_ns` | picoseconds, nanoseconds |
| `_bits`, `_bytes` | bits, bytes |
| `_w` | watts |

Values that a source publishes in GiB are converted to decimal GB when entered, and the published figure is kept verbatim in `notes`.

## Rules

- Blank cell: the value is not published; it loads as `None`.
- Required columns must be non-blank.
- Numbers must parse and be positive.
- Booleans are `true` or `false`.
- `id` values are unique within a file.
- `csp_instances.accelerator_id` must name a row of `accelerators.csv`.
- `accelerators.scaleup_bw_gb_per_s` is the vendor-published aggregate scale-up bandwidth per device; the model uses half of it as the per-direction ring bandwidth.
- `dram_timing` rows whose core timings are labeled "Ramulator Guesstimate" in the source say `guesstimate` in `notes`; derived rows say `derived preset`.
