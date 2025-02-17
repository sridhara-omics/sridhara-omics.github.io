---
title: "HOW-TO setup Google Colab to run Monocle3 (single cell RNA-seq)"
date: '2024-04-08'
layout: post
categories: Genomics
---

  https://colab.research.google.com/    
  
  Requires: units_0.6-6.tar.gz file.

```python
# Load local files to google colab
from google.colab import files
uploaded = files.upload()
```

```python
!pip3 install rpy2
%load_ext rpy2.ipython
```


```python
!apt-get install libgdal-dev
!apt-get install
!apt-get update
!apt-get install libudunits2-dev
!apt-get update
```


```r
%%R
install.packages("units_0.6-6.tar.gz",configure.args='--with-udunits2-include=/usr/include/udunits2')
devtools::install_github("r-spatial/sf")
```


```r
%%R
install.packages("tidyverse")
install.packages("Matrix")
install.packages("cowplot")
if (!requireNamespace("BiocManager", quietly = TRUE))
install.packages("BiocManager")
BiocManager::install()
BiocManager::install(c('BiocGenerics', 'DelayedArray', 'DelayedMatrixStats',
                       'limma', 'S4Vectors', 'SingleCellExperiment',
                       'SummarizedExperiment', 'batchelor', 'Matrix.utils'))
install.packages("devtools")
devtools::install_github('cole-trapnell-lab/leidenbase')
devtools::install_github('cole-trapnell-lab/monocle3')
```


```r
%%R
library(monocle3)
load("your_file.RData")
```

