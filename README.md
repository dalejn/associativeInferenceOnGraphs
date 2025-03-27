# Associative inference on graphs

Adapted from task and model in [Multi-step inference can be improved across the lifespan with individualized memory interventions
](https://osf.io/preprints/psyarxiv/3mhj6_v1).

## Setup

Analysis was run on a M4 Macbook; change the device from 'mps' during analysis if not using Apple Silicon.

```
platform       x86_64-apple-darwin17.0     
arch           x86_64                      
os             darwin17.0                  
system         x86_64, darwin17.0        

language       R                           
version		   R version 4.0.2 (2020-06-22)
```

## Project Organization
```
├── README.md
├── code
│   ├── fig2.R
│   ├── fig3.R
│   ├── fig4.R
│   ├── figs5_6.R
│   └── graphwalk-model
│       ├── README.md
│       ├── graphwalk
│       │   ├── __pycache__
│       │   ├── graphmeta.py
│       │   ├── graphplots.py
│       │   ├── graphtask.py
│       │   ├── graphtrain.py
│       │   ├── learner.py
│       │   ├── main.py
│       │   ├── neuropsychologia_graphtask.py
│       │   ├── neuropsychologia_graphwalk.py
│       │   ├── torch_helpers.py
│       │   └── utils.py
│       ├── parse_meta.ipynb
│       ├── parse_meta_visualizations.ipynb
│       └── torchweights_v2
├── data
│   ├── trained_models_elasticNet_3
│   │   ├── complex_pairs.json
│   │   ├── pair_of_pairs.json
│   │   └── triad_groups.pkl
│   └── trained_models_elasticNet_3_pretrained
└── environment.yaml
```

## Order of scripts
Skip 1 and 2 if you don't want to re-run the task and model. Task structure, simulated data, and models are in /data

1. neuropsychologia_graphtask.py
2. neuropsychologia_graphwalk.py
3. fig2.R
4. fig3.R
5. fig4.R
6. figs5_6.R
