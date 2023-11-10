library(biomaRt)
ensembl_93 = useEnsembl(biomart="genes",dataset="hsapiens_gene_ensembl", version = 93)
query_data = read.csv('data/bmart.csv')
query = unique(query_data$transcript_id)
attributes = c('ensembl_transcript_id','transcript_length', 'ensembl_gene_id', 'start_position', 'end_position', 'strand', 'transcription_start_site', 'transcript_count', 'percentage_gene_gc_content', 'gene_biotype', 'transcript_biotype')
wanted_data = getBM(attributes= attributes ,filters = 'ensembl_transcript_id', values = query, mart = ensembl_93)
#if (sum(is.na(wanted_data))==0) print('QUERY SUCCESSFUL')
if (length(wanted_data$ensembl_transcript_id) != length(query)) {
  set_difference <- query[!(query %in% wanted_data$ensembl_transcript_id)]
  ensembl_93 = useEnsembl(biomart="genes", GRCh = 37, dataset = 'hsapiens_gene_ensembl')
  data = getBM(attributes= attributes, filters = 'ensembl_transcript_id', values = set_difference, mart = ensembl_93)
  wanted_data = rbind(data, wanted_data)
}
write.csv(wanted_data, 'data/biomart_data.csv', row.names = FALSE)

