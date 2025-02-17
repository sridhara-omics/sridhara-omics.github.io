---
layout: post
title:  "Genomics with Snakemake"
date:   '2023-02-26'
categories: Genomics
---

Generating Next-generation sequencing (NGS) data economically is becoming a simpler problem than analyzing the data systematically. There is an increase in the number of tools available to process such data as well (i.e., tools for  converting raw data to proper format, tools for  aligning the reads to the genome, tools for  post-processing the aligned reads to call variants). This results in many challenges to pick the best performing tools, and to put all of these tools in a way that systematically run to generate the desired output, for example, variant calling. [A recent paper describes in detail about the guidelines in clinical genomics](https://www.ncbi.nlm.nih.gov/pubmed/29154853)

Here, I focus on one of the tools built to bring the above mentioned programs together as a pipeline.  

Snakemake is a bioinformatics tool to build pipelines, mostly NGS and related datasets. More info can be found [here] (https://snakemake.readthedocs.io/en/stable/).

Here, I used snakemake to  1. build Bowtie2 index, 2. mapped fastq (raw data i.e., reads) to fasta to generate NGS alignments, 3. and then process output with tools like samtools and bamtools


To run the below pipeline, you need the NGS raw data (i.e., reads in fastq format)  and the fasta file of species of interest.


In the current blog, I used the SRR from one of the projects I worked. More info on this work [here](https://www.ncbi.nlm.nih.gov/pubmed/29152409).
```python
# These SRRs are from the project description from NCBI portal
# https://www.ncbi.nlm.nih.gov/sra/SRX1981073[accn]
# Use sra-tools fasterq-dump to dump the SRRs used in the project
```

To download a fasta file if you know the NCBI accession ID, you can do something like this:
```python
# The fasta used in this analyses is from (name it to k12_profile.fasta)
# https://www.ncbi.nlm.nih.gov/genome?term=NC_000913.3&cmd=DetailsSearch
```

A very first rule i.e., **rule all** is defined with the input that reflects the output of the entire snakemake workflow. In the current post, I start with reads, build index of fasta file, map reads to this built index, then use a threshold to pick high-quality alignments, sort them and finally conver this to BED format. So, the final output is BED format, which should go into the *input* of the _**rule all**_.
```python
# Rule 1: Build rule of all rules?
ecoli_samples = ['SRR5468393']

rule all:
  input: expand("{sample}.bed", sample=ecoli_samples)
```


```python
# Rule 2: build the index file from fasta file
rule bowtie_index_build:
    input:
        "k12_profile.fasta"
    shell:
        "bowtie2-build {input} k12Index/k12_profile"
```


```python
# Rule 3: Map fastq to fasta index
rule bowtie_map:
    input:
        expand("{sample}.fastq", sample=ecoli_samples)
    output:
        expand("{sample}.sam", sample=ecoli_samples)
    shell:
        "bowtie2 -x k12Index/k12_profile -p 4 --very-sensitive --no-unal -U {input} -S {output} -k 1"
```


```python
# Rule 4: Use samtools with q=30 for good alignments i.e., bam with a good cut-off
rule samtools_view:
    input:
        expand("{sample}.sam", sample=ecoli_samples)
    output:
        expand("{sample}.bam", sample=ecoli_samples)
    shell:
        "samtools view -q 30 -bS {input} > {output}"
```


```python
# Rule 5: Sort the bam
rule samtools_sort:
    input:
        expand("{sample}.bam", sample=ecoli_samples)
    output:
        expand("{sample}.Sorted", sample=ecoli_samples)
    shell:
        "samtools sort {input} > {output}"
```


```python
# Rule 6: convert bam to bed format (example)
rule bedtools_BAMtoBED:
    input:
        expand("{sample}.Sorted", sample=ecoli_samples)
    output:
        expand("{sample}.bed", sample=ecoli_samples)
    shell:
        "bedtools bamtobed -i {input} > {output}"
```

Once you put the above commands in a single script (Snakefile), then all you have to do is run "snakemake Snakefile". The same sample script can be found [here as well](https://github.com/viswam78/LASSOprobes/blob/master/Scripts/Snakefile)

If everything goes fine, this is how the output should look like:

```python
Building DAG of jobs...
Using shell: /bin/bash
Provided cores: 1
Rules claiming more threads will be scaled down.
Job counts:
	count	jobs
	1	all
	1	bedtools_BAMtoBED
	1	bowtie_map
	1	samtools_sort
	1	samtools_view
	5

```


```python

[Mon Feb 25 21:05:14 2019]
rule bowtie_map:
    input: SRR5468393.fastq
    output: SRR5468393.sam
    jobid: 4

[Mon Feb 25 21:05:28 2019]
Finished job 4.
1 of 5 steps (20%) done

[Mon Feb 25 21:05:28 2019]
rule samtools_view:
    input: SRR5468393.sam
    output: SRR5468393.bam
    jobid: 3

[Mon Feb 25 21:05:32 2019]
Finished job 3.
2 of 5 steps (40%) done

[Mon Feb 25 21:05:32 2019]
rule samtools_sort:
    input: SRR5468393.bam
    output: SRR5468393.Sorted
    jobid: 2

[Mon Feb 25 21:05:35 2019]
Finished job 2.
3 of 5 steps (60%) done

[Mon Feb 25 21:05:35 2019]
rule bedtools_BAMtoBED:
    input: SRR5468393.Sorted
    output: SRR5468393.bed
    jobid: 1

[Mon Feb 25 21:05:38 2019]
Finished job 1.
4 of 5 steps (80%) done

[Mon Feb 25 21:05:38 2019]
localrule all:
    input: SRR5468393.bed
    jobid: 0

```

You can see that the job finished successfully and the output files formed are sam format, bam format, sorted bam and finally a sample bed file that converts BAM format to BED format.
