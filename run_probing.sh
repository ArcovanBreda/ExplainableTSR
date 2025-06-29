#!/bin/bash

set -e

# clear; parallel -j8 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation {} --idx 1 ::: add cos exp log mul pow sin tan
# -----------------------------------------
# Config: {'operation': 'add', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'cos-tan', 'CTR': False, 'Evaluation_type': 'functional'}
# parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation add --idx ::: 78 80 15 56 94 93 79 59 92 76 66 23 53 12 33 103 87 7 68 31
python NeSymReS/Probing/run_experiment.py --indices 78,80,15,56,94,93,79,59,92,76,66,23,53,12,33,103,87,7,68,31 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_add_correct_ --save results_operation-add_patch_type-resample_CTR_token-cos_CTE-cos-tan_CTR-False_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'add', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'cos-tan', 'CTR': False, 'Evaluation_type': 'model'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation add --idx ::: 114 56 105 58 13 41 75 73 57 44 69 51 86 30 71 10 38 100 7 34
python NeSymReS/Probing/run_experiment.py --indices 114,56,105,58,13,41,75,73,57,44,69,51,86,30,71,10,38,100,7,34 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_add_correct_ --save results_operation-add_patch_type-resample_CTR_token-cos_CTE-cos-tan_CTR-False_Evaluation_type-model

# -----------------------------------------
# Config: {'operation': 'cos', 'patch_type': 'mean', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation cos --idx ::: 60 111 62 10 40 79 77 61 44 76 48 78 29 67 15 37 101 13 85 33
python NeSymReS/Probing/run_experiment.py --indices 60,111,62,10,40,79,77,61,44,76,48,78,29,67,15,37,101,13,85,33 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_cos_correct_ --save results_operation-cos_patch_type-mean_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'cos', 'patch_type': 'mean', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'model'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation cos --idx ::: 87 94 11 58 109 57 56 40 101 89 66 100 71 49 105 34 38 26 77 8
python NeSymReS/Probing/run_experiment.py --indices 87,94,11,58,109,57,56,40,101,89,66,100,71,49,105,34,38,26,77,8 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_cos_correct_ --save results_operation-cos_patch_type-mean_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-model

# -----------------------------------------
# Config: {'operation': 'cos', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation cos --idx ::: 91 96 14 60 111 48 104 92 115 38 88 66 35 100 26 28 5 69 4 52
python NeSymReS/Probing/run_experiment.py --indices 91,96,14,60,111,48,104,92,115,38,88,66,35,100,26,28,5,69,4,52 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_cos_correct_ --save results_operation-cos_patch_type-resample_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'cos', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'model'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation cos --idx ::: 79 91 15 58 111 104 82 64 102 55 84 32 51 5 20 103 86 4 67 19
python NeSymReS/Probing/run_experiment.py --indices 79,91,15,58,111,104,82,64,102,55,84,32,51,5,20,103,86,4,67,19 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_cos_correct_ --save results_operation-cos_patch_type-resample_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-model

# -----------------------------------------
# Config: {'operation': 'cos', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': True, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation cos --idx ::: 88 101 13 64 111 48 106 89 115 37 92 62 39 94 21 23 8 75 6 57
python NeSymReS/Probing/run_experiment.py --indices 88,101,13,64,111,48,106,89,115,37,92,62,39,94,21,23,8,75,6,57 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_cos_correct_ --save results_operation-cos_patch_type-resample_CTR_token-cos_CTE-sin-tan_CTR-True_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'exp', 'patch_type': 'mean', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation exp --idx ::: 69 73 12 46 82 80 70 55 78 60 71 28 64 13 34 105 102 7 79 32
python NeSymReS/Probing/run_experiment.py --indices 69,73,12,46,82,80,70,55,78,60,71,28,64,13,34,105,102,7,79,32 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_exp_correct_ --save results_operation-exp_patch_type-mean_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'exp', 'patch_type': 'mean', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'model'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation exp --idx ::: 61 110 70 15 48 80 79 64 55 78 38 73 21 66 12 32 103 7 84 29
python NeSymReS/Probing/run_experiment.py --indices 61,110,70,15,48,80,79,64,55,78,38,73,21,66,12,32,103,7,84,29 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_exp_correct_ --save results_operation-exp_patch_type-mean_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-model

# -----------------------------------------
# Config: {'operation': 'log', 'patch_type': 'mean', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation log --idx ::: 80 84 21 60 96 94 82 74 92 78 61 106 15 48 10 28 104 85 7 64
python NeSymReS/Probing/run_experiment.py --indices 80,84,21,60,96,94,82,74,92,78,61,106,15,48,10,28,104,85,7,64 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_618_log_correct_ --save results_operation-log_patch_type-mean_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'log', 'patch_type': 'mean', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'model'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation log --idx ::: 79 84 19 60 100 98 80 71 97 75 64 106 107 17 49 11 31 104 85 8
python NeSymReS/Probing/run_experiment.py --indices 79,84,19,60,100,98,80,71,97,75,64,106,107,17,49,11,31,104,85,8 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_618_log_correct_ --save results_operation-log_patch_type-mean_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-model

# -----------------------------------------
# Config: {'operation': 'log', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation log --idx ::: 74 80 13 49 102 97 76 58 96 68 84 106 26 77 7 38 104 5 85 34
python NeSymReS/Probing/run_experiment.py --indices 74,80,13,49,102,97,76,58,96,68,84,106,26,77,7,38,104,5,85,34 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_618_log_correct_ --save results_operation-log_patch_type-resample_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'log', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'model'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation log --idx ::: 79 84 19 60 101 98 80 71 97 55 38 49 12 28 110 85 7 65 88 24
python NeSymReS/Probing/run_experiment.py --indices 79,84,19,60,101,98,80,71,97,55,38,49,12,28,110,85,7,65,88,24 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_618_log_correct_ --save results_operation-log_patch_type-resample_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-model

# -----------------------------------------
# Config: {'operation': 'log', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': True, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation log --idx ::: 110 47 95 112 49 12 35 59 58 48 53 84 67 101 31 86 14 52 100 7
python NeSymReS/Probing/run_experiment.py --indices 110,47,95,112,49,12,35,59,58,48,53,84,67,101,31,86,14,52,100,7 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_618_log_correct_ --save results_operation-log_patch_type-resample_CTR_token-cos_CTE-sin-tan_CTR-True_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'mul', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation mul --idx ::: 59 114 61 12 40 78 77 60 43 74 49 82 32 69 14 39 104 7 85 35
python NeSymReS/Probing/run_experiment.py --indices 59,114,61,12,40,78,77,60,43,74,49,82,32,69,14,39,104,7,85,35 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_mul_correct_ --save results_operation-mul_patch_type-resample_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'mul', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'model'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation mul --idx ::: 114 52 101 56 13 39 73 62 55 43 70 53 87 31 82 12 41 102 7 35
python NeSymReS/Probing/run_experiment.py --indices 114,52,101,56,13,39,73,62,55,43,70,53,87,31,82,12,41,102,7,35 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_mul_correct_ --save results_operation-mul_patch_type-resample_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-model

# -----------------------------------------
# Config: {'operation': 'pow', 'patch_type': 'mean', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation pow --idx ::: 67 76 10 43 101 95 74 50 94 59 86 58 84 29 66 103 23 87 65 85
python NeSymReS/Probing/run_experiment.py --indices 67,76,10,43,101,95,74,50,94,59,86,58,84,29,66,103,23,87,65,85 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_pow_correct_ --save results_operation-pow_patch_type-mean_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'pow', 'patch_type': 'mean', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'model'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation pow --idx ::: 67 73 10 43 97 94 70 51 92 59 89 49 87 29 66 100 23 65 91 93
python NeSymReS/Probing/run_experiment.py --indices 67,73,10,43,97,94,70,51,92,59,89,49,87,29,66,100,23,65,91,93 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_pow_correct_ --save results_operation-pow_patch_type-mean_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-model

# -----------------------------------------
# Config: {'operation': 'sin', 'patch_type': 'mean', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation sin --idx ::: 57 60 10 37 76 74 59 41 73 47 82 29 70 16 46 106 101 13 84 38
python NeSymReS/Probing/run_experiment.py --indices 57,60,10,37,76,74,59,41,73,47,82,29,70,16,46,106,101,13,84,38 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_sin_correct_ --save results_operation-sin_patch_type-mean_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'sin', 'patch_type': 'mean', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'model'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation sin --idx ::: 74 76 5 41 92 89 75 56 82 62 69 107 28 61 17 34 104 91 15 71
python NeSymReS/Probing/run_experiment.py --indices 74,76,5,41,92,89,75,56,82,62,69,107,28,61,17,34,104,91,15,71 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_sin_correct_ --save results_operation-sin_patch_type-mean_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-model

# -----------------------------------------
# Config: {'operation': 'sin', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation sin --idx ::: 55 110 57 12 35 74 68 56 40 61 52 84 29 73 17 48 104 7 88 46
python NeSymReS/Probing/run_experiment.py --indices 55,110,57,12,35,74,68,56,40,61,52,84,29,73,17,48,104,7,88,46 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_sin_correct_ --save results_operation-sin_patch_type-resample_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'sin', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'model'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation sin --idx ::: 52 110 56 12 35 75 74 55 40 61 53 86 32 71 19 48 104 7 88 42
python NeSymReS/Probing/run_experiment.py --indices 52,110,56,12,35,75,74,55,40,61,53,86,32,71,19,48,104,7,88,42 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_sin_correct_ --save results_operation-sin_patch_type-resample_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-model

# -----------------------------------------
# Config: {'operation': 'sin', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': True, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation sin --idx ::: 110 48 95 112 55 12 37 61 60 53 50 76 62 101 33 83 25 49 100 7
python NeSymReS/Probing/run_experiment.py --indices 110,48,95,112,55,12,37,61,60,53,50,76,62,101,33,83,25,49,100,7 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_sin_correct_ --save results_operation-sin_patch_type-resample_CTR_token-cos_CTE-sin-tan_CTR-True_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'tan', 'patch_type': 'mean', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation tan --idx ::: 114 51 97 56 8 32 69 62 52 40 76 55 93 34 84 23 47 105 14 42
python NeSymReS/Probing/run_experiment.py --indices 114,51,97,56,8,32,69,62,52,40,76,55,93,34,84,23,47,105,14,42 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_tan_correct_ --save results_operation-tan_patch_type-mean_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'tan', 'patch_type': 'mean', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'model'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation tan --idx ::: 111 56 100 58 12 40 74 73 57 44 68 50 89 28 71 13 37 98 8 32
python NeSymReS/Probing/run_experiment.py --indices 111,56,100,58,12,40,74,73,57,44,68,50,89,28,71,13,37,98,8,32 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_tan_correct_ --save results_operation-tan_patch_type-mean_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-model

# -----------------------------------------
# Config: {'operation': 'tan', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation tan --idx ::: 88 93 15 59 111 42 98 89 115 35 103 67 47 105 19 26 5 69 4 62
python NeSymReS/Probing/run_experiment.py --indices 88,93,15,59,111,42,98,89,115,35,103,67,47,105,19,26,5,69,4,62 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_tan_correct_ --save results_operation-tan_patch_type-resample_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-functional

# -----------------------------------------
# Config: {'operation': 'tan', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'model'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation tan --idx ::: 114 58 101 61 13 43 76 74 60 52 66 46 86 29 68 11 34 102 7 32
python NeSymReS/Probing/run_experiment.py --indices 114,58,101,61,13,43,76,74,60,52,66,46,86,29,68,11,34,102,7,32 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_tan_correct_ --save results_operation-tan_patch_type-resample_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-model

# -----------------------------------------
# Config: {'operation': 'tan', 'patch_type': 'resample', 'CTR_token': 'cos', 'CTE': 'sin-tan', 'CTR': False, 'Evaluation_type': 'functional'}
parallel -j10 --ungroup python NeSymReS/DataGeneration/ProbeDatasetGeneration.py --operation tan --idx ::: 110 44 94 112 55 12 34 61 60 46 51 82 62 102 32 84 14 50 101 7
python NeSymReS/Probing/run_experiment.py --indices 110,44,94,112,55,12,34,61,60,46,51,82,62,102,32,84,14,50,101,7 --datapath /home/arco/Downloads/Master/MscThesis/ExplainableDSR/data/Arco/Datasets/Probing/cached_values_1000_tan_correct_ --save results_operation-tan_patch_type-resample_CTR_token-cos_CTE-sin-tan_CTR-False_Evaluation_type-functional

