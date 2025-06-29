# Explaining the Explainer: Understanding the Inner Workings of Transformer-based Symbolic Regression Models

This repository contains the code and resources for my Master's thesis, *Explaining the Explainer: Understanding the Inner Workings of Transformer-based Symbolic Regression Models*, conducted at the university of amsterdam supervised by Erman Acar.

## Abstract

> Understanding the internal mechanisms of deep learning models remains a central challenge in interpretability research. This thesis investigates the opaque behaviour of transformer-based symbolic regression models through the lens of mechanistic interpretability, an underexplored field in this context. Focusing on circuit-level analysis, we identify circuits (minimal subgraphs) responsible for specific operators within the model. We introduce PATCHES, a novel circuit discovery method, which yields smaller and more correct circuits than existing approaches. Our study identifies 28 single-token circuits across eight symbolic operations and two multi-token circuits, comparing common patching methods, evaluation strategies, and direct logit attribution (DLA) with cumulative circuit discovery. We find that mean patching with performance-based evaluation most effectively identifies correct circuits and generally yields the smallest circuits. In contrast, DLA cannot be reliably evaluated, and we caution against its use for circuit discovery. Finally, we probe our circuits and find that high probing scores do not significantly correlate with circuit membership, reinforcing concerns that probing reflects correlation rather than causal involvement. This work contributes both methodological tools and advances a more robust circuit discovery pipeline.

## Technical Contributions

- **Systematic evaluation of circuit discovery techniques**:We compare **mean patching** vs. **resample patching**, distinguish between **model faithfulness** and **functional faithfulness**, and contrast **direct logit attribution** with full **circuit discovery**.
- **A reproducible attribution patching pipeline**:We provide a clear implementation of attribution patching using `NNsight`, integrating best practices into a cohesive and adaptable pipeline. This includes formal definitions of **faithfulness**, **completeness**, and **minimality**, as well as a new test for **uniqueness**. The pipeline is model-agnostic and easily adapted across domains.
- **PATCHES: a novel method for circuit discovery**:
  We introduce **PATCHES** (*Probabilistic Algorithm for Tuning Circuits through Heuristic Evolution and Search*), a CMA-ES-based method that discovers smaller and more correct circuits than traditional iterative patching.

---

## Getting Started

```bash
git clone https://github.com/ArcovanBreda/ExplainableTSR.git
cd ExplainableTSR
pip install -r requirements.txt
```

---

## Recreating Results

All thesis code is located in the root folder or in `NeSymRes/`. Scripts in `scripts/` and `src/` are from the original [NeSymReS paper](https://arxiv.org/pdf/2106.06427) by Biggio et al.

---

### **Generate Datasets**

   Run `NeSymReS/DataGeneration/data_generation.py` for both constant-on and constant-off modes

---

### **Evaluate Model Performance**

1. Execute:

```
python NeSymReS/ModelPerformance/ModelPerformance.py  
python NeSymReS/ModelPerformance/ModelPerformanceFAI.py  
```

2. Visualize results with:

```
jupyter notebook NeSymReS/ModelPerformance/model_performance_for_thesis.ipynb 
```

---

### **Discover Circuits**

#### Step 1: Generate Patch Datasets (Example: `sin` operation)

```
# Generate mean patches
python NeSymReS/DataGeneration/mean_patching_datagen.py

# Generate resample patches
python NeSymReS/DataGeneration/Resample_Patching_dataset2.0.py \
    --n_corr_equations 1000 \
    --number_of_points 200 \
    --character_to_include sin \
    --character_to_change_to cos \
    --characters_to_exclude cos tan \
    --top 3

# Cache resample data
python NeSymReS/DataGeneration/resample_caching.py \
    --operation sin \
    --control cos \
    --subset TRAIN \
    --num_points 200 \
    --max_datapoints 100 \
    --data_root YOUR_DATA_ROOT \
    --save_root YOUR_SAVE_ROOT
```

#### Step 2: Run CMA-ES Optimization

```
python NeSymReS/Faithfulness/cma-es.py \
    --operation sin \
    --num_points 200 \
    --max_iterations 117 \
    --random_seed 42 \
    --max_samples 100 \
    --cma_max_evals 10000 \
    --num_workers 8 \
    --patch_type mean \
    --patch_type_subset TRAIN \
    --patch_type_CTR cos \
    --evaluation_type functional
```

##### Full Argument Specifications for CMA-ES (PATCHES):

| Argument                | Type | Default          | Description                                                  |
| ----------------------- | ---- | ---------------- | ------------------------------------------------------------ |
| `--operation`         | str  | `"sin"`        | Target operation (e.g.,`"sin"`, `"cos"`)                 |
| `--num_points`        | int  | `200`          | Number of evaluation points per equation                     |
| `--max_iterations`    | int  | `117`          | Search space dimension (typically # of attention heads)      |
| `--random_seed`       | int  | `None`         | Random seed for reproducibility                              |
| `--max_samples`       | int  | `100`          | Max equations to evaluate                                    |
| `--cma_max_evals`     | int  | `10000`        | Max CMA-ES evaluations                                       |
| `--num_workers`       | int  | `8`            | Parallel worker threads                                      |
| `--patch_type`        | str  | `"mean"`       | Patch type (`mean` or `resample`)                        |
| `--patch_type_subset` | str  | `"TRAIN"`      | Data subset (`TRAIN` or `same_decode`)                   |
| `--patch_type_CTR`    | str  | `"cos"`        | Token to replace during patching                             |
| `--evaluation_type`   | str  | `"functional"` | Evaluation metric (`functional` or `model` faithfulness) |

#### Step 3: Run Iterative-Patching for minimality

```
python NeSymReS/Faithfulness/IterativePatching.py
    --operation sin \
    --num_points 200 \
    --max_iterations 117 \
    --excluded_heads 1 3 5 \
    --random_seed 42 \
    --max_samples 100 \
    --patch_type mean \
    --patch_type_CTR cos \
    --evaluation_type functional \
    --patch_type_subset TRAIN
```

#### Step 4: Evaluate

1. Save your found circuit in `circuit_config.json`
2. Run evaluation:

```
jupyter notebook NeSymReS/Faithfulness/Evaluation.ipynb
```

---

### **Reproduce Probing**

1. Execute `run_probing.sh`:

```
chmod +x run_probing.sh  # Make executable if needed
./run_probing.sh
```

2. Evaluate results:

```
jupyter notebook NeSymReS/Probing/probing_tests.ipynb
```

---

### **Reproduce DLA (Direct Logit Attribution)**

1. Execute patching scripts:

```
python NeSymReS/Patching/MeanPatching.py
python NeSymReS/Patching/ResamplePatching.py
```

2. Visualize results:

```
jupyter notebook NeSymReS/Patching/DLA_results.ipynb
```

---

## Acknowledgments

Thank you for your interest in this research. If you have any questions or would like to discuss this work further, please don't hesitate to reach out:

- Email: [arcovanbreda@gmail.com](mailto:arcovanbreda@gmail.com)
- LinkedIn: [Arco van Breda](https://www.linkedin.com/in/arcovanbreda/)
- Thesis Advisor: [Erman Acar](https://www.linkedin.com/in/erman-acar/)

---
