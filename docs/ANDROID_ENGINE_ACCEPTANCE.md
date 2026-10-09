# PhotoForge Android LiteRT engine acceptance

Priority: image-to-image photo editing. Secondary: text-to-image generation. Both offline.

## Required integration contract (not yet implemented)
1. Native Android host must use compatible LiteRT CompiledModel GPU API; generic TFLite OpenCL delegates are not interchangeable.
2. Run staged graph loading: text encoder -> editing kce_* or generation kc_* -> VAE. Release each graph before loading next; ensure cleanup on cancellation/exceptions.
3. Verify all 21 graph files and tokenizer assets against trusted SHA256 manifest before execution. The metadata API alone does not provide a complete trusted hash manifest.
4. Reject incomplete models; check disk availability before fresh download; use temporary files and atomic rename after verification. Do not report success from a placeholder image.
5. Measure native heap, graphics allocations, device peak RAM, thermal throttling and elapsed time on Galaxy S25 Ultra; test airplane mode.
6. Save edited and generated images via Android MediaStore; do not overwrite source photo.
7. Show actionable errors for insufficient disk, failed checksum, unsupported GPU delegate, allocation failure and user cancellation.

## Current verification status
- Python offline integrity checker and regression tests: code present; GitHub Actions run must be inspected separately.
- Android engine wired to app: NO.
- Real 21-graph loading: NO.
- Real edit output: NO.
- Real generated output: NO.
- S25 Ultra peak GPU/RAM measured: NO.

Do not substitute metadata checks or synthetic fixtures for actual device inference.
