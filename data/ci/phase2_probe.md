# Phase 2 feasibility probe

## ClinicalTrials.gov v2 (oracle candidate: trials)
- intervention: pembrolizumab: **3001** registered studies
- intervention: metformin: **3137** registered studies
- intervention: semaglutide: **796** registered studies
- condition: breast cancer: **16983** registered studies
- cond breast cancer, phase 3 only: **1564** registered studies

One study's protocolSection layout:
```json
{
 "identificationModule": [
  "acronym",
  "briefTitle",
  "nctId",
  "officialTitle",
  "orgStudyIdInfo",
  "organization",
  "secondaryIdInfos"
 ],
 "statusModule": [
  "completionDateStruct",
  "expandedAccessInfo",
  "lastUpdatePostDateStruct",
  "lastUpdateSubmitDate",
  "overallStatus",
  "primaryCompletionDateStruct",
  "resultsFirstPostDateStruct",
  "resultsFirstSubmitDate",
  "resultsFirstSubmitQcDate",
  "startDateStruct",
  "statusVerifiedDate",
  "studyFirstPostDateStruct",
  "studyFirstSubmitDate",
  "studyFirstSubmitQcDate"
 ],
 "sponsorCollaboratorsModule": [
  "leadSponsor",
  "responsibleParty"
 ],
 "oversightModule": [
  "oversightHasDmc"
 ],
 "descriptionModule": [
  "briefSummary"
 ],
 "conditionsModule": [
  "conditions",
  "keywords"
 ],
 "designModule": [
  "designInfo",
  "enrollmentInfo",
  "phases",
  "studyType"
 ],
 "armsInterventionsModule": [
  "armGroups",
  "interventions"
 ],
 "outcomesModule": [
  "primaryOutcomes",
  "secondaryOutcomes"
 ],
 "eligibilityModule": [
  "eligibilityCriteria",
  "healthyVolunteers",
  "minimumAge",
  "sex",
  "stdAges"
 ],
 "contactsLocationsModule": [
  "locations",
  "overallOfficials"
 ],
 "referencesModule": [
  "references"
 ]
}
```

## OpenAlex (oracle candidate: citations)
- 'BERT: Pre-training of Deep Bidirectional Transformers' -> matched 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding' (2018): **47071** citations — https://openalex.org/W2896457183
- 'Attention Is All You Need' -> matched 'Attention Is All You Need' (2017): **122093** citations — https://openalex.org/W2626778328
- 'Deep Residual Learning for Image Recognition' -> matched 'Deep Residual Learning for Image Recognition' (2016): **237350** citations — https://openalex.org/W2194775991

Work record fields: ['abstract_inverted_index', 'apc_list', 'apc_paid', 'authorships', 'awards', 'best_oa_location', 'biblio', 'citation_normalized_percentile', 'cited_by_count', 'cited_by_percentile_year', 'concepts', 'content_urls', 'corresponding_author_ids', 'corresponding_institution_ids', 'countries_distinct_count', 'counts_by_year', 'created_date', 'display_name', 'doi', 'funders', 'fwci', 'has_content', 'has_fulltext', 'id', 'ids', 'indexed_in', 'institutions', 'institutions_distinct_count', 'is_paratext', 'is_retracted', 'is_xpac', 'keywords', 'language', 'locations', 'locations_count', 'mesh', 'open_access', 'primary_location', 'primary_topic', 'publication_date', 'publication_year', 'referenced_works', 'referenced_works_count', 'related_works', 'relevance_score', 'study_designs', 'sustainable_development_goals', 'title', 'topics', 'type', 'updated_date', 'volume']
