# BGE evidence and review scripts

These scripts preserve the analysis entry points used for the 100-machine results.
Run `remote.py` and `host_run_fp16_retest.py` on a host with SSH/Docker access.
Run the collection and graph scripts inside `hejun-cann9-bge` against the
original project root `/export/home/ext.hejun23/workspace/bge-opt-100`.
They are research/review tools, not the production inference entry point.
Do not reuse old job names or claim that invoking a collector reruns the NPU job.
