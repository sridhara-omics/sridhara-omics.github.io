---
title: "Infectious Disease Related Drugs - Exploratory analyses"
date: '2024-05-11'
layout: post
output:
  html_document:
    df_print: paged
  pdf_document: default
  word_document: default
categories: Genomics
---

The data used in this blog is taken from 2 resources: [1] drug repurposing hub tab of the CLUE.IO (https://clue.io/repurposing#download-data) and [2] cell surface proteins (CSPs) from the SurfaceomeDB.  

CLUE.IO: The Drug Repurposing Hub is a curated and annotated collection of FDA-approved drugs, clinical trial drugs, and pre-clinical tool compounds with a companion information resource. Current dataset is downloaded on May 11, 2020.


```r
# required packages
rm(list = ls())
library(tidyverse)
```


Out of 6798 total drugs, 2427 are already launched, while 458 are in Phase 3 of the clinical trials.


```r
# Drugs in different phases of clinical trials
df_phases <- drug_gene %>% group_by(clinical_phase) %>% summarise(count=n())
head(df_phases)
```

```
## # A tibble: 6 x 2
##   clinical_phase  count
##   <fct>           <int>
## 1 Launched         2427
## 2 Phase 1           566
## 3 Phase 1/Phase 2    85
## 4 Phase 2           813
## 5 Phase 2/Phase 3    44
## 6 Phase 3           458
```

Out of 6798 total drugs, 424 drugs are classified in "infectious disease". Most of the infectious disease related drugs (423/424) are already launched. 4576 drugs are still in different phases of the clinical trials.

```r
# Drugs in different disease areas
df_disease_area <- drug_gene %>% group_by(disease_area) %>% summarise(count=n())
df_disease_area_order <- df_disease_area[order(-df_disease_area$count),]
head(df_disease_area_order)
```

```
## # A tibble: 6 x 2
##   disease_area           count
##   <fct>                  <int>
## 1 ""                      4576
## 2 "infectious disease"     424
## 3 "neurology/psychiatry"   346
## 4 "cardiology"             205
## 5 "gastroenterology"       124
## 6 "endocrinology"          122
```

```r
# Drugs in different disease areas
df_disease_area_launched <- drug_gene %>% group_by(disease_area) %>% filter(clinical_phase == "Launched") %>% summarise(count=n())
df_disease_area_launched_order <- df_disease_area_launched[order(-df_disease_area_launched$count),]
head(df_disease_area_launched_order)
```

```
## # A tibble: 6 x 2
##   disease_area           count
##   <fct>                  <int>
## 1 "infectious disease"     423
## 2 "neurology/psychiatry"   346
## 3 ""                       213
## 4 "cardiology"             204
## 5 "gastroenterology"       124
## 6 "endocrinology"          122
```
Total number of gene targets in the entire list is 2183.


```r
# Distribution of genes (unique and all)
all_genes <- unlist(strsplit(as.character(drug_gene$target),split='|',fixed=TRUE))
unique_genes <- unique(all_genes)
```

If we focus only on the infectious disease drug targets, there are 149 unique gene targets.

```r
# Distribution of genes (unique and all)
drug_gene_inf_dis <- subset(drug_gene, disease_area == "infectious disease")
all_genes_inf_dis <- unlist(strsplit(as.character(drug_gene_inf_dis$target),split='|',fixed=TRUE))
unique_genes_inf_dis <- unique(all_genes_inf_dis)
unique_genes_inf_dis
```

```
##   [1] "GABBR1"  "GABBR2"  "MMP12"   "HIF1A"   "PNP"     "TUBA1A"  "TUBB"   
##   [8] "TUBB4B"  "ADRA1A"  "ADRA2A"  "HNMT"    "CYP3A4"  "ATP1A1"  "CYP2B6"
##  [15] "HTR3A"   "HTR3B"   "MLNR"    "IDE"     "SCN10A"  "DAO"     "HRSP12"
##  [22] "PRDX5"   "RAB9A"   "HSPA1A"  "HSPB1"   "CMA1"    "CTSA"    "CTSF"   
##  [29] "CTSK"    "CTSL"    "CTSS"    "SQLE"    "TP53"    "PON1"    "FASN"   
##  [36] "MRGPRX1" "DHFR"    "KCNN4"   "NR1I2"   "NR1I3"   "TRPM2"   "TRPM4"  
##  [43] "TRPM8"   "CYP3A43" "CYP3A5"  "CYP3A7"  "PGR"     "GRIN1"   "PNLIP"  
##  [50] "KCNN1"   "KCNN3"   "ALOX5"   "PTGS1"   "ACHE"    "POU2F2"  "UGT1A1"
##  [57] "DRD2"    "DRD3"    "DPEP1"   "NPY1R"   "NPY2R"   "TRPV5"   "CYP1A2"
##  [64] "CYP2C19" "CYP2C9"  "CYP2D6"  "CYP51A1" "CYP19A1" "CYP2J2"  "PPARA"  
##  [71] "PTGER2"  "TOP2A"   "ABCB1"   "ALB"     "KCNH2"   "SLC47A1" "DNMT1"  
##  [78] "METAP2"  "KRT12"   "GLUD1"   "SDHD"    "TYR"     "TLR7"    "TLR9"   
##  [85] "CYP17A1" "CYP2C8"  "CYP2E1"  "CHRNA7"  "P2RX7"   "GABRB1"  "GLRA1"  
##  [92] "GLRA2"   "GLRA3"   "GLRB"    "MAOA"    "MAOB"    "CA12"    "CA14"   
##  [99] "CA2"     "CA4"     "CA6"     "CA9"     "CCR5"    "PLA2G1B" "SLC22A6"
## [106] "CYP2A6"  "STAT3"   "CES1"    "NEU1"    "NEU2"    "TRDMT1"  "SCN1A"  
## [113] "GABRB3"  "IGF1R"   "TUBA4A"  "CCR2"    "AR"      "GP9"     "KCNB2"  
## [120] "SLC29A4" "ADK"     "ENPP1"   "IMPDH1"  "IMPDH2"  "NT5C2"   "P4HB"   
## [127] "SLCO1A2" "SLCO1B1" "SLCO1B3" "SLCO2B1" "PTPN6"   "DHPS"    "F2"     
## [134] "FSHR"    "P2RY1"   "P2RY11"  "P2RY13"  "P2RY2"   "PLA2G2A" "RYR1"   
## [141] "RYR2"    "SIRT5"   "ALPPL2"  "CHRNA3"  "OXCT1"   "TRPA1"   "TYMS"   
## [148] "ADA"     "TERT"
```
I used the uniprot mapping tool to map the UniProt IDs in the Surfaceome DB to gene names. In total, there are 1247 unique cell surface receptors that are identified by mass-spec and other proteomics based assays.

```r
# cell surface receptors
csr <- read.csv("C:\\Users\\Viswa\\Downloads\\GSE148729\\csr_proteinID_geneName.txt", sep = "\t")
csr_genenames <- toupper(csr$To)
head(csr)
```

```
##     From       To
## 1 A2A699 Fam171a2
## 2 A2A863    Itgb4
## 3 A2A8L5    Ptprf
## 4 A2AFS3 Kiaa1324
## 5 A2AJN7  Slc4a11
## 6 A2AJQ3  Dpy19l4
```
Out of 1247 cell surface receptors, 20 receptors were classified as the drug targets for the infectious disease.

```r
inf_dis_csrs <- unique_genes_inf_dis[unique_genes_inf_dis %in% csr_genenames]
inf_dis_csrs
```

```
##  [1] "GABBR1"  "GABBR2"  "ADRA2A"  "ATP1A1"  "CMA1"    "CTSA"    "CTSF"   
##  [8] "CTSL"    "PON1"    "GRIN1"   "DPEP1"   "KCNH2"   "P2RX7"   "CA12"   
## [15] "CA4"     "GABRB3"  "IGF1R"   "GP9"     "SLC29A4" "ENPP1"
```
Subsetting the approved drug metadata to focus only on these 20 cell surface receptors:


```r
gene2drug <- function(gene, df){
  #df_subset <- subset(df, target %in% gene)
  gene <- unlist(gene)
  df_subset <- df[grep(gene, df$target),]
  df_subset <- data.frame(df_subset)
  return(df_subset)
}

geneList <- as.list(inf_dis_csrs)
df_csr_inf_dis_subset <- lapply(geneList, gene2drug, df = drug_gene_inf_dis)
cell_surface_infectious_disease_drugs <- do.call(rbind, df_csr_inf_dis_subset)
csr_inf_drugs <- unique(cell_surface_infectious_disease_drugs)
csr_inf_drugs[c(1,3,4,6)]
```

```
##                       pert_iname
## 35                     abamectin
## 330                      amitraz
## 5801                  talipexole
## 466                   artemether
## 1387                  ciclopirox
## 3516                lumefantrine
## 934                   boceprevir
## 5903                  telaprevir
## 1203                   cefazolin
## 1648             cycloserine-(D)
## 1981                   doripenem
## 2212       erythromycin-estolate
## 2213 erythromycin-ethylsuccinate
## 3150                  ivermectin
## 3599                    mafenide
## 4761                  piperazine
## 4828             podophyllotoxin
## 5028                     quinine
## 5130                   ribavirin
##                                                         moa
## 35                          benzodiazepine receptor agonist
## 330                             adrenergic receptor agonist
## 5801  adrenergic receptor agonist|dopamine receptor agonist
## 466                                      antimalarial agent
## 1387                           membrane integrity inhibitor
## 3516                                     antimalarial agent
## 934                                           HCV inhibitor
## 5903                                          HCV inhibitor
## 1203                bacterial cell wall synthesis inhibitor
## 1648                bacterial cell wall synthesis inhibitor
## 1981                bacterial cell wall synthesis inhibitor
## 2212              bacterial 50S ribosomal subunit inhibitor
## 2213  cytochrome P450 inhibitor|protein synthesis inhibitor
## 3150                        benzodiazepine receptor agonist
## 3599                           carbonic anhydrase inhibitor
## 4761                        benzodiazepine receptor agonist
## 4828 microtubule inhibitor|tubulin polymerization inhibitor
## 5028                  hemozoin biocrystallization inhibitor
## 5130                                              antiviral
##                                           target
## 35                                 GABBR1|GABBR2
## 330                                ADRA1A|ADRA2A
## 5801                           ADRA2A|DRD2|HTR3A
## 466                                       ATP1A1
## 1387                                      ATP1A1
## 3516                                      ATP1A1
## 934                CMA1|CTSA|CTSF|CTSK|CTSL|CTSS
## 5903                                    CTSA|PGR
## 1203                                        PON1
## 1648                                       GRIN1
## 1981                                       DPEP1
## 2212 ABCB1|ALB|CYP3A4|CYP51A1|KCNH2|MLNR|SLC47A1
## 2213 ABCB1|ALB|CYP3A4|CYP51A1|KCNH2|MLNR|SLC47A1
## 3150                                CHRNA7|P2RX7
## 3599                   CA12|CA14|CA2|CA4|CA6|CA9
## 4761                                      GABRB3
## 4828                     IGF1R|TOP2A|TUBA4A|TUBB
## 5028                     GP9|KCNB2|KCNN4|SLC29A4
## 5130               ADK|ENPP1|IMPDH1|IMPDH2|NT5C2
##                                                                                                                                indication
## 35                                                                                                             gastrointestinal parasites
## 330                                                                                                               generalized demodicosis
## 5801                                                                                                                      genitial herpes
## 466                                                                                                                               malaria
## 1387                                                                                                                        onychomycosis
## 3516                                                                                                                              malaria
## 934                                                                                                                           hepatitis C
## 5903                                                                                                                          hepatitis C
## 1203            urinary tract infections|skin infections|bacterial septicemia|endocarditis|surgical prophylaxis|bone and joint infections
## 1648                                                                                                            tuberculosis|tuberculosis
## 1981                                                                   intra-abdominal infections|urinary tract infections|pyelonephritis
## 2212 listeria|respiratory tract infections|skin infections|syphilis|amebiasis|pelvic inflammatory disease|chlamydia|diphtheria|erythrasma
## 2213 listeria|respiratory tract infections|skin infections|syphilis|amebiasis|pelvic inflammatory disease|chlamydia|diphtheria|erythrasma
## 3150                                                             gastrointestinal roundworms|lungworms|cattle grubs|mites|lice|horn flies
## 3599                                                                                                 first-aid antibiotic|skin infections
## 4761                                                                                                          gastrointestinal roundworms
## 4828                                                                                                                        genital warts
## 5028                                                                                                                              malaria
## 5130                                                                                                                          hepatitis C
```
