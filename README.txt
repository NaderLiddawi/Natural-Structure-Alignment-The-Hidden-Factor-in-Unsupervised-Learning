README.txt for Assignment 3  
===============================================================================

Assignment 3 – Unsupervised Learning and Dimensionality Reduction   


-------------------------------------------------------------------------------

Overview:
---------
This repository contains the implementation for **Assignment 3** of **CS-7641 (Spring 2025)**.  
The main script, `main.py`, coordinates the execution of unsupervised learning algorithms and dimensionality reduction techniques. It includes:

- **Clustering algorithms**: K-Means and Expectation Maximization (Gaussian Mixture Models)
- **Dimensionality reduction methods**: PCA, ICA, and Random Projection
- **Combined analysis**: Evaluation of clustering + reduction pipelines
- **Neural network**: Performance evaluation with reduced feature sets

**Required inputs**:
- `marketing_campaign.csv`
- `spotify-2023.csv`  
These files must be in the same directory as `main.py`.

Note: Instructions are written for Linux but apply equally to Windows with minor changes (e.g., virtual environment activation syntax).

-------------------------------------------------------------------------------

Directory Structure:
--------------------
The repository has the following layout:

- `README.txt` ............. This file with setup and usage instructions  
- `main.py` ................ Primary script implementing all tasks  
- `marketing_campaign.csv` . Dataset for customer marketing analysis  
- `spotify-2023.csv` ....... Dataset for audio/music features  
- `requirements.txt` ....... List of Python package dependencies (optional)  
- `results/` ............... Auto-generated directory containing output files, organized into:
    - `clustering/`
    - `dim_reduction/`
    - `combined/`
    - `neural_net/`
    - `extra/`

-------------------------------------------------------------------------------

Dependencies and Setup:
------------------------
Python version: **3.8 or higher** (Python 3.11 recommended)

Required packages:

- `numpy` ............ Numerical computing  
- `pandas` ........... Data manipulation  
- `scipy` ............ Statistical methods  
- `matplotlib` ....... Plotting  
- `seaborn` .......... Statistical visualization  
- `scikit-learn` ..... ML algorithms & evaluation metrics  
- `tabulate` ......... Console table formatting  
- `kneed` ............ Elbow method for optimal cluster selection

**Option 1: Install individually**  
pip install numpy pandas scipy matplotlib seaborn scikit-learn tabulate kneed

**Option 2: Install from requirements.txt**  


-------------------------------------------------------------------------------

Data Files:
-----------
Ensure the following files are in the same folder as `main.py`:

1. `marketing_campaign.csv` - Demographic and purchasing info  
2. `spotify-2023.csv`       - Track features from Spotify's 2023 dataset

-------------------------------------------------------------------------------

Running the Code:
---------------------
To execute the code and reproduce the results, follow these detailed steps:

   1. **Clone the Repository:**
      - Open a command prompt (on Windows) or terminal (on Linux).
      - Run:
            git clone https://github.com/NaderLiddawi/Natural-Structure-Alignment-The-Hidden-Factor-in-Unsupervised-Learning.git

   2. **Set Up the Python Environment:**
      - Create a virtual environment:
            python -m venv env
      - Activate the virtual environment:
            env\Scripts\activate   (on Windows)
         or
            source env/bin/activate   (on Linux)

   3. **Install Dependencies:**
      - Install all required packages:
            pip install -r requirements.txt
         (or install packages individually as listed above).

   4. **Verify Data Files:**
      - Confirm that `marketing_campaign.csv` and `spotify-2023.csv` are located in the same directory as `main.py`.

   5. **Run the Script:**
      - Execute the main Python script:
            python main.py


- Outputs include plots, metrics, clustering analyses, and neural network performance.  
- Results will appear both in the console and under the `results/` folder.

-------------------------------------------------------------------------------

Reproducibility and Determinism:
--------------------------------
- All algorithms use a fixed **random seed (RANDOM_SEED=42)** to guarantee reproducibility.
- Every pipeline (clustering, reduction, neural net) logs outputs and parameters.
- Inline comments and modular function definitions facilitate transparency and auditability.
- Execution times may vary due to different hardware quality, background processes and other factors that may affect execution times with each run and for each computer.

-------------------------------------------------------------------------------

Overleaf Project and Final Commit:
----------------------------------
To inspect the final Overleaf report and development notes:

Overleaf (READ ONLY):  
https://www.overleaf.com/read/dhbhqfcmrqpd#50ed54

-------------------------------------------------------------------------------

Additional Notes:
-----------------

- Each computational step is clearly documented in both code and this README.
- Output files are organized into relevant subdirectories under `/results` for clarity.
- The implementation includes an **extra credit** section on **non-linear manifold learning**.
- For bugs or reproducibility concerns, check commit history or contact the maintainer.

This project adheres to principles of scientific reproducibility and transparent reporting.

===============================================================================

