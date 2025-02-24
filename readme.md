## Setup

1. Create the conda environment:
   '''
   bash
   conda env create -f environment.yml
   conda activate my_project_env

2. Run the main script:
'''
python main.py



project/
├── main.py                     # Main entry point: loads data, processes domains, and builds models
├── config.py                   # Configuration file (e.g., file paths, constants)
├── data_loader.py              # Module to load and merge raw datasets and add PHQ-4 categorization
├── feature_extractor.py        # Module to extract feature names from the labeled CSV file
├── feature_engineering.py      # Module for cleaning data and adding calculated columns
├── domain_processor.py         # Module to process each interaction label (Leisure, Me Time, Phone Time, Sleep Time, Social Time)
├── model_builder.py            # Module for building and training personalized neural network models per UID
├── utils.py                    # Utility functions (e.g., YAML configuration loader)
├── environment.yml             # Conda environment file listing required packages and Python version
└── README.md                   # Documentation with setup instructions and an overview of the project
