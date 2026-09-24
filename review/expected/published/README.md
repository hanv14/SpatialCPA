# expected/published — slot for the lab's published row

Empty on purpose. Copy the published STARmap v18 row here from the machine that
produced it:

    expected/published/spatialcpav18_gen/starmap_visual_cortex/paper_2_4_6/
        metrics.json  prediction.h5  method_log.txt  resources.json

then `make manifest` and commit. `make starmap-row` compares against this row
automatically once `metrics.json` exists (otherwise against expected/cpu-verified/).
Before trusting it, check that `prediction.h5`'s `uns/method_params` has
`edit_weight: 0.0` and `ground_blend_flow: 1.0` (REVIEW_NOTES §1).
