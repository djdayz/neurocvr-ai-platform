# NeuroCVR-AI Experiment Report

This report summarises the synthetic GLM benchmark sweep for NeuroCVR-AI.

The sweep evaluates how cerebrovascular reactivity magnitude and delay recovery change across temporal contrast-to-noise ratio conditions.

## Experiment setup

The benchmark sweep varies:

- tCNR noise level
- random seed

For each run, the pipeline:

1. Generates synthetic ground-truth CVR magnitude and delay maps.
2. Simulates BOLD-fMRI data from an ETCO2 stimulus.
3. Fits a voxelwise GLM with delay search.
4. Estimates CVR magnitude and delay maps.
5. Compares estimates against synthetic ground truth.
6. Logs parameters and metrics to MLflow.

## Key metrics

The tracked metrics are:

- CVR magnitude RMSE
- CVR magnitude MAE
- CVR magnitude bias
- CVR magnitude PCC
- Delay RMSE
- Delay MAE
- Delay bias
- Delay PCC

## Best CVR magnitude recovery

Best mean CVR RMSE occurred at:

- tCNR: 5.0000
- mean CVR RMSE: 0.0317
- mean CVR PCC: 0.9965

## Best delay recovery

Best mean delay RMSE occurred at:

- tCNR: 5.0000
- mean delay RMSE: 0.3871 s
- mean delay PCC: 0.9843

## Summary table

| tcnr | cvr_rmse_mean | cvr_rmse_std | cvr_mae_mean | cvr_mae_std | cvr_bias_mean | cvr_bias_std | cvr_pcc_mean | cvr_pcc_std | delay_rmse_mean | delay_rmse_std | delay_mae_mean | delay_mae_std | delay_bias_mean | delay_bias_std | delay_pcc_mean | delay_pcc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.3107296208305024 | 0.018339566803854 | 0.2286845862905105 | 0.0181318803179813 | 0.04096535911256 | 0.0357889734301067 | 0.7708935232909101 | 0.0487000146721417 | 2.6652121473335137 | 0.0186683122846461 | 2.063757016879679 | 0.0457078719757996 | 0.1239335658532482 | 0.3163601500169626 | 0.3751809706169165 | 0.1029174791834464 |
| 1.0 | 0.1558880826250071 | 0.0069087528139852 | 0.1146653012453673 | 0.0081339236351994 | 0.0125509520314989 | 0.0174601832599246 | 0.9235781206358376 | 0.0116194949364145 | 1.82414876764268 | 0.1816057063109512 | 1.311861623509785 | 0.1077502069096516 | 0.0239335658532482 | 0.1539532916554809 | 0.6727083562104679 | 0.0930196619192368 |
| 2.0 | 0.0780882176015696 | 0.0039002676083777 | 0.056978350596509 | 0.0042975355164028 | 0.0026884608135213 | 0.0090204423272065 | 0.9792118463098312 | 0.0024420357603124 | 0.8427175786611881 | 0.1239269237153543 | 0.633675225514261 | 0.0848491580800303 | 0.0183780102976926 | 0.0956796415604954 | 0.9258729523387212 | 0.0226334813531281 |
| 5.0 | 0.0316514604494915 | 0.0016391433869192 | 0.0230783082832433 | 0.0015793951925173 | 0.0003491763862894 | 0.003608233437519 | 0.9965000007995628 | 0.0003320806854287 | 0.387112061010551 | 0.0341372106779955 | 0.3230799178977101 | 0.0384282752804066 | -0.0072190436932007 | 0.0227491838925617 | 0.984274185257978 | 0.0020220300690504 |

## Figures

### CVR magnitude RMSE vs tCNR

![CVR magnitude RMSE vs tCNR](cvr_rmse_vs_tcnr.png)

### Delay RMSE vs tCNR

![Delay RMSE vs tCNR](delay_rmse_vs_tcnr.png)

## Interpretation

Higher tCNR generally improves CVR magnitude and delay recovery because the simulated BOLD response is less dominated by noise.

Low-tCNR runs are expected to show larger errors and lower correlations, making them useful stress tests for the estimation pipeline.

## Engineering relevance

This report demonstrates:

- experiment tracking with MLflow
- repeatable benchmark sweeps
- CSV export of run-level and summary metrics
- automated report generation
- portfolio-ready visualisation of model behaviour

## Safety and limitations

This project is a research and portfolio prototype.

The reported metrics are based on synthetic data with known ground truth. They should not be interpreted as clinical validation.

This software is not a diagnostic tool and must not be used for patient care.
