# Preserved packaging qualification failure

The first isolated deployment passed file-SHA verification on both hosts, then
failed importing the scientific engine in synthetic qualification (actual child
exit 1 on both hosts). No real dataset was loaded, no official seed training
started, and no scientific checkpoint was produced.

Cause: `simple_lab_test/common/pathing.py` identifies the project root by the
presence of `models`, `utils`, and `sample_data`. The first bundle placed its
admitted input under `inputs`, omitting `sample_data`. Root discovery therefore
selected the ancestor server checkout and imported its older
`utils/magnitude_pipeline.py`; this module referenced unavailable
`models.RMTPPs`. The failure is a deployment isolation bug, not a model or data
failure and not evidence of a server connectivity failure.

The v1 remote directories, archive, contract, check receipts, actual failure
exit/log bytes and all corresponding local transport records are retained.
The correction uses a separate `execution_v2` contract, bundle and remote roots.
It places the same immutable Train/Validation bytes under `sample_data`, adds
fail-closed bootstrap/import-source checks, and tests the actual extracted
bundle outside the original repository before native qualification.

Repeating prelaunch qualification after this packaging correction does not
repeat any scientific fit. The three requested fresh fits remain unstarted
until v2 local review and both native qualification gates pass.
