# FarmaCorp: A Spanish–English Pharmacological Domain Corpus

FarmaCorp is a bilingual macrocorpus (Spanish–English) in the **pharmacological domain**, designed to support research and professional practice in translation, interpreting, and natural language processing (NLP).  
It was developed by the Lexytrad Research Group (University of Málaga) through a semi-automatic compilation and expert supervision process.

## Corpus composition

FarmaCorp is structured into three complementary subcorpora:

- **Comparable corpus (FarmaCorp_comp):**  
  Collections of domain-specific original texts in Spanish and English, including drug information leaflets, research articles, regulations, and institutional documents.  
  - FC_ES: 36,450 Spanish texts (~170M tokens)  
  - FC_EN: 9,786 English texts (~17M tokens)

- **Parallel corpus (FarmaCorp_par):**  
  Spanish–English aligned texts from MEDLINE and Ken Pharma, consisting of research articles, medical encyclopedia entries, and drug leaflets.  
  - 2,544 aligned documents  
  - ~1.4M Spanish tokens, ~1.3M English tokens

- **Multimodal corpus (FarmaCorp_mul):**  
  76 YouTube videos (83h 45m) with transcripts in both Spanish and English, subdivided by diatopic varieties (e.g., Iberian Spanish, American English, British English).

## Preprocessing and alignment

- Texts were **extracted** from online resources (HTML, PDF, multimedia) using a semi-automatic pipeline.  
- Documents were **normalized** to plain text, cleaned, and organized by language.  
- For the parallel corpus, **sentence alignment** was carried out using the Gale–Church algorithm and manually revised.  
- For the multimodal corpus, **automatic transcripts** were retrieved and manually curated.  

## Machine translation evaluation

FarmaCorp was applied to evaluate **state-of-the-art machine translation (MT)** systems in the biomedical domain.  
A **stratified group k-fold cross-validation** procedure was used (5 folds, with 80/20 train/validation splits).  

Evaluated models included:
- Helsinki-NLP/opus-mt-es-en  
- facebook/m2m100_1.2B  
- facebook/m2m100_418M  
- facebook/nllb-200-distilled-600M  
- Helsinki-NLP/opus-mt-es-en-finetuned (domain-adapted)

**Metrics:** BLEU, chrF, and TER.  
Results demonstrated that the **finetuned model outperformed all baselines**, confirming the importance of specialized domain corpora.

## Availability

All corpora are publicly available on Zenodo:  
[FarmaCorp on Zenodo](https://zenodo.org/records/17233158)


