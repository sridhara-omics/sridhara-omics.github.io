---
layout: post
title:  "Flux Balance Analyses with COBRApy"
date:   2023-02-23
categories: Systems Biology
---

Understanding biology by analyzing high-dimensional -OMICS data sets is a general trend these days. This high-dimensional data could be a result from an experimental run on a Next-Generation Sequencing (NGS) platform or a mass-spectrometer platform or a  similar high-throughput technique. In case of NGS, generally the measured molecules are DNA/RNA. In case of mass-spec, the measured _ionized_ molecules could be proteins, metabolites or lipids. 

This trend of gather-analyze-report data in biological space is increasing at an exponential space given the lowering cost of sequencing genomes using NGS technologies.  

Systematically analyzing such datasets is critical to draw meaningful observations. Moreover, a technique to integrate the diverse omics datasets (e.g., transcriptomics and proteomics data for the same biological sample) is also a new trend that is increasing. The advantage with such integration is to infer insights using the information from transcript and protein profiles. 

Metabolic network models provide an interesting way to overlay this diverse information i.e., data sets emerging from different technologies a.k.a NGS based transcripts, mass-spec baed protein abundances, and 13C mass-spec based metabolic fluxes. 

Before going into integrating these datasets (possibly in future posts), it is essential to understand the metabolic network models from informatics point of view. These days, the metabolic network models are built computationally using the genome sequence and homology based methods and further refined using other resources, e.g., manual curation using previous publications. The models built are then stored in json, xml, SBML formats for further computational analyses. More information on such an effort in _E. coli_ and other organisms can be found [here](https://www.sri.com/work/projects/ecocyc). 

Here I focus on analyzing one such _E. coli_ metabolic network iAF1260 model using a mathematical technique called Flux Balance Analysis (FBA). FBA is an approach to computationally analyze the flow of metabolites in a metabolic network ([paper on FBA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3108565/)). Such methods of using genome-scale metabolic network model to analyze the flow of metabolites generally are called [COBRA based methods](https://opencobra.github.io/).

Previously, in a different project at [Wilke lab](https://wilkelab.org/), we used matlab version of COBRA. Here, I use COBRApy, a python version of COBRA. Most of the information on the COBRApy modules used can be found [here](https://cobrapy.readthedocs.io/en/stable/) 

I used some of the modules mentioned above to fit my research needs and specifically here, I show how to 1. read the metabolic network model, 2. generate optimal fluxes, 3. knock-out reactions, 4. change growth media, 5. report data using the jupyter notebook. 


Let's import modules needed. cobra is a module needed to analyze different metabolic network models.
```python
from __future__ import print_function
import cobra
import os
from os.path import join
```

The model we'll analyze is iAF1260.
```python
sbml_path = join("Data","iAF1260.xml.gz")
print(sbml_path)

```

    Data\iAF1260.xml.gz
    

The format of the model is SBML. SBML stands for systems biology markup model. More info [here](https://en.wikipedia.org/wiki/SBML)
```python
iAF1260_ecoli_model = cobra.io.read_sbml_model(sbml_path)
```

The model.optimize() call is the crux of the analyses. Given the objective function, this optimize find a feasible solution i.e., solution here is a vector of in-silico metabolic fluxes.
```python
# Here the objective function is biomass and optimize function calculates the fluxes to get the max biomass (this can be changed)
iAF1260_ecoli_model.optimize()
```




<strong><em>Optimal</em> solution with objective value 0.737</strong><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fluxes</th>
      <th>reduced_costs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12DGR120tipp</th>
      <td>0.000000</td>
      <td>-0.010031</td>
    </tr>
    <tr>
      <th>12DGR140tipp</th>
      <td>0.000000</td>
      <td>-0.010031</td>
    </tr>
    <tr>
      <th>12DGR141tipp</th>
      <td>0.000000</td>
      <td>-0.016049</td>
    </tr>
    <tr>
      <th>12DGR160tipp</th>
      <td>0.000000</td>
      <td>-0.010031</td>
    </tr>
    <tr>
      <th>12DGR161tipp</th>
      <td>0.000000</td>
      <td>-0.018055</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ZN2abcpp</th>
      <td>0.000000</td>
      <td>-0.008025</td>
    </tr>
    <tr>
      <th>ZN2t3pp</th>
      <td>0.000000</td>
      <td>-0.002006</td>
    </tr>
    <tr>
      <th>ZN2tpp</th>
      <td>0.002327</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ZNabcpp</th>
      <td>0.000000</td>
      <td>-0.008025</td>
    </tr>
    <tr>
      <th>Zn2tex</th>
      <td>0.002327</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>2382 rows × 2 columns</p>
</div>



The function model.summary() prints out the exchange fluxes, along with the optimized solution. In this case, objective function we're trying to maximize is the model growth (i.e., Biomass) and looks like the value is 0.737.
```python
iAF1260_ecoli_model.summary()
```

    IN FLUXES            OUT FLUXES    OBJECTIVES
    -------------------  ------------  ----------------------
    o2_e       16.3      h2o_e  37.2   BIOMASS_Ec_i...  0.737
    glc__D_e    8        co2_e  17.8
    nh4_e       7.94     h_e     6.77
    pi_e        0.708
    so4_e       0.184
    k_e         0.131
    mg2_e       0.00582
    fe2_e       0.00556
    fe3_e       0.00523
    ca2_e       0.00349
    cl_e        0.00349
    cobalt2_e   0.00233
    cu2_e       0.00233
    mn2_e       0.00233
    mobd_e      0.00233
    zn2_e       0.00233
    

We can also knowck-out reactions and re-run the optimize function and calculate the new fluxes.
```python
# remove nitrogen source and look at Biomass objective function
# here the only nitrogen source seems to be nh4_e.
iAF1260_ecoli_model.reactions.NH4tex.knock_out()
```


```python
iAF1260_ecoli_model.optimize()
```




<strong><em>Optimal</em> solution with objective value 0.000</strong><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fluxes</th>
      <th>reduced_costs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12DGR120tipp</th>
      <td>0.000000e+00</td>
      <td>3.330669e-16</td>
    </tr>
    <tr>
      <th>12DGR140tipp</th>
      <td>0.000000e+00</td>
      <td>3.330669e-16</td>
    </tr>
    <tr>
      <th>12DGR141tipp</th>
      <td>0.000000e+00</td>
      <td>6.106227e-16</td>
    </tr>
    <tr>
      <th>12DGR160tipp</th>
      <td>0.000000e+00</td>
      <td>3.330669e-16</td>
    </tr>
    <tr>
      <th>12DGR161tipp</th>
      <td>0.000000e+00</td>
      <td>6.106227e-16</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ZN2abcpp</th>
      <td>0.000000e+00</td>
      <td>2.220446e-16</td>
    </tr>
    <tr>
      <th>ZN2t3pp</th>
      <td>0.000000e+00</td>
      <td>5.551115e-17</td>
    </tr>
    <tr>
      <th>ZN2tpp</th>
      <td>7.906134e-18</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>ZNabcpp</th>
      <td>0.000000e+00</td>
      <td>2.220446e-16</td>
    </tr>
    <tr>
      <th>Zn2tex</th>
      <td>7.906134e-18</td>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>
<p>2382 rows × 2 columns</p>
</div>




```python
# get the original model 
iAF1260_ecoli_model = cobra.io.read_sbml_model(sbml_path)


```

We can also control the concentration of the media to mimic real-situations. The values adjusted could reflect generally used media such as minimal media etc.
```python
# change glucose source fluxes to different values and see how it affects the objective function
iAF1260_ecoli_model.medium
```




    {'EX_ca2_e': 999999.0,
     'EX_cbl1_e': 0.01,
     'EX_cl_e': 999999.0,
     'EX_co2_e': 999999.0,
     'EX_cobalt2_e': 999999.0,
     'EX_cu2_e': 999999.0,
     'EX_fe2_e': 999999.0,
     'EX_fe3_e': 999999.0,
     'EX_glc__D_e': 8.0,
     'EX_h2o_e': 999999.0,
     'EX_h_e': 999999.0,
     'EX_k_e': 999999.0,
     'EX_mg2_e': 999999.0,
     'EX_mn2_e': 999999.0,
     'EX_mobd_e': 999999.0,
     'EX_na1_e': 999999.0,
     'EX_nh4_e': 999999.0,
     'EX_o2_e': 18.5,
     'EX_pi_e': 999999.0,
     'EX_so4_e': 999999.0,
     'EX_tungs_e': 999999.0,
     'EX_zn2_e': 999999.0}



This is one of the simpler ways to change the concentration. In this specific case, initially the glucose concentration in the media is 8. Here, we read all the media related fluxes to a vector, change the vector and initialize the media with this new vector. Below, we change the concentration from 8 to 20.
```python
# currently EX_glc__D_e is at 8, change it to 20 (Instead of knocking out NH4tex, this is another way of changing the source)
# copy the mediums, change the values and put the medium back in the model (there might be simpler way)
medium = iAF1260_ecoli_model.medium
medium["EX_glc__D_e"] = 20
iAF1260_ecoli_model.medium = medium
```


```python
iAF1260_ecoli_model.optimize()
```




<strong><em>Optimal</em> solution with objective value 1.218</strong><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fluxes</th>
      <th>reduced_costs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12DGR120tipp</th>
      <td>0.000000</td>
      <td>-0.024236</td>
    </tr>
    <tr>
      <th>12DGR140tipp</th>
      <td>0.000000</td>
      <td>-0.024236</td>
    </tr>
    <tr>
      <th>12DGR141tipp</th>
      <td>0.000000</td>
      <td>-0.043625</td>
    </tr>
    <tr>
      <th>12DGR160tipp</th>
      <td>0.000000</td>
      <td>-0.024236</td>
    </tr>
    <tr>
      <th>12DGR161tipp</th>
      <td>0.000000</td>
      <td>-0.043625</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ZN2abcpp</th>
      <td>0.000000</td>
      <td>-0.019389</td>
    </tr>
    <tr>
      <th>ZN2t3pp</th>
      <td>0.000000</td>
      <td>-0.004847</td>
    </tr>
    <tr>
      <th>ZN2tpp</th>
      <td>0.003846</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ZNabcpp</th>
      <td>0.000000</td>
      <td>-0.019389</td>
    </tr>
    <tr>
      <th>Zn2tex</th>
      <td>0.003846</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>2382 rows × 2 columns</p>
</div>



Recalculated fluxes with the new *glucose concentration*.
```python
iAF1260_ecoli_model.summary()
```

    IN FLUXES            OUT FLUXES           OBJECTIVES
    -------------------  -------------------  ---------------------
    glc__D_e   20        h_e       53         BIOMASS_Ec_i...  1.22
    o2_e       18.5      h2o_e     41.6
    nh4_e      13.1      for_e     23.1
    pi_e        1.17     ac_e      18.7
    so4_e       0.305    co2_e      9.53
    k_e         0.216    glyclt_e   0.000815
    mg2_e       0.00961
    fe2_e       0.0092
    fe3_e       0.00865
    ca2_e       0.00577
    cl_e        0.00577
    cobalt2_e   0.00385
    cu2_e       0.00385
    mn2_e       0.00385
    mobd_e      0.00385
    zn2_e       0.00385
    

If we plan to suck out glucose from the _E. coli_ model?
```python
# Repeat the above by changing it to -20 (see the negative sign)
medium["EX_glc__D_e"] = -20
iAF1260_ecoli_model.medium = medium
```

Optimization seems to result in infeasible solution. More information on how FBA analyses is done, please refer to the [original paper by Orth J. D. et. al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3108565/)
```python
iAF1260_ecoli_model.optimize()
```

    cobra\util\solver.py:416 UserWarning: solver status is 'infeasible'
    




<strong><em>infeasible</em> solution</strong>



Let's take the glucose concentration back to 8.
```python
# Back to original value of 8
medium["EX_glc__D_e"] = 8
iAF1260_ecoli_model.medium = medium
```


```python
iAF1260_ecoli_model.optimize()
```




<strong><em>Optimal</em> solution with objective value 0.737</strong><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fluxes</th>
      <th>reduced_costs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12DGR120tipp</th>
      <td>0.000000</td>
      <td>-0.010031</td>
    </tr>
    <tr>
      <th>12DGR140tipp</th>
      <td>0.000000</td>
      <td>-0.010031</td>
    </tr>
    <tr>
      <th>12DGR141tipp</th>
      <td>0.000000</td>
      <td>-0.018055</td>
    </tr>
    <tr>
      <th>12DGR160tipp</th>
      <td>0.000000</td>
      <td>-0.010031</td>
    </tr>
    <tr>
      <th>12DGR161tipp</th>
      <td>0.000000</td>
      <td>-0.018055</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ZN2abcpp</th>
      <td>0.000000</td>
      <td>-0.008025</td>
    </tr>
    <tr>
      <th>ZN2t3pp</th>
      <td>0.000000</td>
      <td>-0.002006</td>
    </tr>
    <tr>
      <th>ZN2tpp</th>
      <td>0.002327</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ZNabcpp</th>
      <td>0.000000</td>
      <td>-0.008025</td>
    </tr>
    <tr>
      <th>Zn2tex</th>
      <td>0.002327</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>2382 rows × 2 columns</p>
</div>


To get an idea of energy production:

```python
# Get a sense of energy production and related features
iAF1260_ecoli_model.metabolites.atp_c.summary()

```

    PRODUCING REACTIONS -- ATP C10H12N5O13P3 (atp_c)
    ------------------------------------------------
    %      FLUX  RXN ID      REACTION
    ---  ------  ----------  --------------------------------------------------
    72%  52      ATPS4rpp    adp_c + 4.0 h_p + pi_c <=> atp_c + h2o_c + 3.0 h_c
    18%  13.1    PGK         3pg_c + atp_c <=> 13dpg_c + adp_c
    5%    3.34   SUCOAS      atp_c + coa_c + succ_c <=> adp_c + pi_c + succoa_c
    4%    2.72   PPK         atp_c + pi_c <=> adp_c + ppi_c
    1%    1.03   PYK         adp_c + h_c + pep_c --> atp_c + pyr_c
    
    CONSUMING REACTIONS -- ATP C10H12N5O13P3 (atp_c)
    ------------------------------------------------
    %      FLUX  RXN ID      REACTION
    ---  ------  ----------  --------------------------------------------------
    61%  44.2    BIOMASS...  0.000223 10fthf_c + 0.000223 2ohph_c + 0.5137 a...
    12%   8.39   ATPM        atp_c + h2o_c --> adp_c + h_c + pi_c
    9%    6.19   PFK         atp_c + f6p_c --> adp_c + fdp_c + h_c
    3%    1.99   NDPK1       atp_c + gdp_c <=> adp_c + gtp_c
    2%    1.78   ACCOAC      accoa_c + atp_c + hco3_c --> adp_c + h_c + malc...
    2%    1.33   GLNS        atp_c + glu__L_c + nh4_c --> adp_c + gln__L_c +...
    1%    0.788  ASPK        asp__L_c + atp_c <=> 4pasp_c + adp_c
    1%    0.687  R15BPK      atp_c + r15bp_c --> adp_c + prpp_c
    1%    0.687  R1PK        atp_c + r1p_c --> adp_c + h_c + r15bp_c
    

Initially we used the default optimization function i.e., biomass maximization. With FBA, you can also change this objective function to mimic your study. For example, you can find new set of fluxes when maximizing ATPM.
```python
# Instead of maximizing biomass, we can change the objective function to maximize ATPM
iAF1260_ecoli_model.objective = "ATPM"
iAF1260_ecoli_model.reactions.get_by_id("ATPM").upper_bound


```




    8.39




```python
iAF1260_ecoli_model.reactions.get_by_id("ATPM").lower_bound
```




    8.39



Most likely, the lower and upper bound of such reactions would be set to experimentally defined value (manual curation). In order to find the max/min, you should set the upper and lower bound values to reasonable values.
```python
# Looks like the ATPM upper bound and lower bound is fixed at 8.39.
# If we run the model now, the optimum calculated will be same.
# INSTEAD, change the upper and lower bounds to different values and see the optimum with objective function of ATPM

iAF1260_ecoli_model.reactions.get_by_id("ATPM").upper_bound = 1000
iAF1260_ecoli_model.reactions.get_by_id("ATPM").lower_bound = -1000
```


```python
iAF1260_ecoli_model.reactions.get_by_id("ATPM").upper_bound
```




    1000




```python
iAF1260_ecoli_model.reactions.get_by_id("ATPM").lower_bound
```




    -1000




```python
iAF1260_ecoli_model.optimize()
```




<strong><em>Optimal</em> solution with objective value 95.813</strong><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fluxes</th>
      <th>reduced_costs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12DGR120tipp</th>
      <td>0.0</td>
      <td>-2.5</td>
    </tr>
    <tr>
      <th>12DGR140tipp</th>
      <td>0.0</td>
      <td>-2.5</td>
    </tr>
    <tr>
      <th>12DGR141tipp</th>
      <td>0.0</td>
      <td>-4.5</td>
    </tr>
    <tr>
      <th>12DGR160tipp</th>
      <td>0.0</td>
      <td>-2.5</td>
    </tr>
    <tr>
      <th>12DGR161tipp</th>
      <td>0.0</td>
      <td>-4.5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ZN2abcpp</th>
      <td>0.0</td>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>ZN2t3pp</th>
      <td>0.0</td>
      <td>-0.5</td>
    </tr>
    <tr>
      <th>ZN2tpp</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ZNabcpp</th>
      <td>0.0</td>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>Zn2tex</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2382 rows × 2 columns</p>
</div>



The ATPM seems to change significantly. This might not reflect true scenario and might be artifact. However the benefits of using models is to quickly simulate to see if there is anything unusual before designing such experiments.
```python
# Looks like the ATPM value went from 9 all the way to 95 (based on constraints on other reactions)
# Summary of atp reactions with this new objective function is
iAF1260_ecoli_model.metabolites.atp_c.summary()
```

    PRODUCING REACTIONS -- ATP C10H12N5O13P3 (atp_c)
    ------------------------------------------------
    %      FLUX  RXN ID    REACTION
    ---  ------  --------  --------------------------------------------------
    61%   63.8   ATPS4rpp  adp_c + 4.0 h_p + pi_c <=> atp_c + h2o_c + 3.0 h_c
    15%   16     PGK       3pg_c + atp_c <=> 13dpg_c + adp_c
    14%   14.8   ACKr      ac_c + atp_c <=> actp_c + adp_c
    8%     8     PYK       adp_c + h_c + pep_c --> atp_c + pyr_c
    1%     1.25  SUCOAS    atp_c + coa_c + succ_c <=> adp_c + pi_c + succoa_c
    
    CONSUMING REACTIONS -- ATP C10H12N5O13P3 (atp_c)
    ------------------------------------------------
    %      FLUX  RXN ID    REACTION
    ---  ------  --------  --------------------------------------------------
    92%   95.8   ATPM      atp_c + h2o_c <=> adp_c + h_c + pi_c
    8%     8     PFK       atp_c + f6p_c --> adp_c + fdp_c + h_c
    

Feel free to contact me if you need more information or if you want to set-up such a study! I hope this is helpful.
```python

```
