## pathing issues
## make sure installation works on ubuntu
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("biomaRt")

library(biomaRt)
ensembl_93 = useEnsembl(biomart="genes",dataset="hsapiens_gene_ensembl", version = 93)
query_data = read.csv('data/bmart.csv')
query = unique(query_data$transcript_id)
wanted_data = getBM(attributes=c('ensembl_transcript_id', 'transcript_biotype', 'transcript_length', 'transcript_tsl'),filters = 'ensembl_transcript_id', values = query, mart = ensembl_93)
if (sum(is.na(wanted_data))==0) print('QUERY SUCCESSFUL')

write.csv(wanted_data, 'data/biomart_data.csv', row.names = FALSE)

