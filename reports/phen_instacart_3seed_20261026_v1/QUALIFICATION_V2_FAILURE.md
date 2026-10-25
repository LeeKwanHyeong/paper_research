# Preserved native qualification result, execution_v2

The corrected bundle bootstrap passed on both hosts. On 5080, check, synthetic
and capacity checks all exited 0. Batch128 × sequence64 synthetic capacity
reported 431,595,520 peak allocated bytes and 490,733,568 peak reserved bytes.
These are qualification values, not final measured training cost.

On 5090, synthetic qualification exited 1 on the recorded-duration CUDA path:
`nvrtc: error: failed to open libnvrtc-builtins.so.13.0`. The file is already
installed under the ai_env `site-packages/nvidia/cu13/lib` directory. The
deployment launcher had set `LD_LIBRARY_PATH` for 5080 only.

No scientific fit or checkpoint was started on either host. All v2 files and
actual exit records are preserved. A new execution_v3 contract binds each
host's process-local CUDA library path, and the qualification and launch
commands use the same binding. No package installation, system environment
change, network/auth change, or alteration of another job is involved.
