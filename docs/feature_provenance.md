# Feature Provenance Example

The preprocessing stack exports a feature map that shows how raw schema fields
become transformed model inputs.

For model tuning examples, see [Config Examples](config_examples.md).

Available columns:

- `final_feature`
- `source_features`
- `source_columns`
- `transformation_path`
- `provenance_depth`
- `feature_type`

## Example Feature Map

| final_feature | source_columns | transformation_path | provenance_depth |
| --- | --- | --- | --- |
| `comorbidities__diabetes` | `["comorbidities"]` | `["raw", "multi_value_expand"]` | `2` |
| `heart_rate_bpm_mean_7d` | `["patient_id", "recorded_at", "heart_rate_bpm"]` | `["raw", "time_series_engineer"]` | `2` |
| `gender_female` | `["gender"]` | `["raw", "one_hot_encode"]` | `2` |
| `pca_1` | `[...]` | `["raw", "scaling", "pca"]` | `3` |

## Notes

- `source_columns` lists the exact raw fields used to build the transformed
  feature.
- `transformation_path` describes the transformation stages in order.
- `provenance_depth` counts the number of stages in the path.
- For PCA rows, `source_columns` is aggregated across all upstream features that
  feed the compressed component.
