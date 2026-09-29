# Catalog sources

Every row cites its own `source_url`; this file summarizes provenance and caveats.

## accelerators.csv

Vendor datasheets, product pages, and whitepapers (NVIDIA, AMD, Intel, Google Cloud, Apple).
Compute figures are dense: where a vendor publishes only "with sparsity" values (H100, H200, L4, HGX B200, GB200), the catalog stores half and the notes quote the published number.
Known ambiguities are recorded per row, for example B200 bandwidth (8.0 TB/s on the DGX B200 page versus 7.75 TB/s implied by the HGX summary; 8.0 is used), GH200 LPDDR5X bandwidth (384 versus 500 GB/s), and Gaudi 3 BF16 (1678 TFLOPS in the 2025 whitepaper versus 1835 at launch; 1678 is used).
FP4 dense peaks: HGX B200 (72 PFLOPS dense per 8 GPUs), GB200 NVL72 (1440 PFLOPS sparse per 72 GPUs, dense is half), MI355X (80.5 PFLOPS MXFP4 per 8 GPUs).
`apple_m2.memory_gb` is the measured 8 GB configuration.

## memory_standards.csv

JEDEC press releases for HBM (JESD235 family), HBM3 (JESD238), HBM4 (JESD270-4), and vendor pages for HBM3E, which has no separate JEDEC standard.
Computed values (for example HBM3E 1254 GB/s = 9.8 Gb/s x 1024 bits / 8) are labeled in notes.

## dram_timing.csv

`hbm2_2000` and `hbm3_6400` come from Ramulator 2 (MIT license) at commit `72427a1bba3771564c4fb0e494ba02242fd1eaa7`, files `python/ramulator/dram/hbm2.py` and `hbm3.py`.
Their core timings are labeled "Ramulator Guesstimate" in the source; refresh and turnaround timings follow JEDEC-cited formulas in the same files.
`hbm2e_3200` and `hbm3e_9600` are derived presets: nanosecond timings of the base preset are held constant, cycle counts are recomputed as `ceil(n_base * tck_base / tck_new)`, and `tCCD_L`, `tRTW`, `tRFC`, and `tREFI` are recomputed with the base source's formulas.
They are sensitivity points, not vendor presets.

## models.csv

`config.json` files on Hugging Face (ungated mirrors for Llama models) and parameter counts from the Hugging Face safetensors metadata API.

## tiers.csv

NVIDIA Grace Hopper page for NVLink-C2C (900 GB/s total, 450 GB/s per direction); PCI-SIG link rates for PCIe Gen4 and Gen5 x16 (per direction, after 128b/130b encoding).

## csp_instances.csv

AWS EC2 accelerated-computing instance documentation, Google Cloud accelerator-optimized machine documentation, and Microsoft Azure ND-series documentation.
Providers mix GB and GiB; the published figure and unit are kept in notes, and host memory published in GiB is converted to GB.
