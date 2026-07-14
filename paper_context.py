PAPER_TEXT = """
Title: Multimodal Data Integration Improves Disease Risk Prediction in the UK Biobank

Abstract:
Family health history is an important component to assess risk for common chronic diseases. The integration of electronic health records and genetic data offers great potential to improve disease risk prediction by capturing both clinical and genetic risk factors. We present ALIGATEHR-Gen, a graph attention network that integrates multimodal patient data including genetic information, diagnosis codes, and demographics, along with external medical ontology knowledge. ALIGATEHR-Gen constructs unified patient representations by incorporating genetically inferred first-degree relationships and disease ontology embeddings to enhance disease risk prediction. We evaluate the predictive performance of ALIGATEHR-Gen across 118 diseases in the UK Biobank and demonstrate that it outperforms state-of-the-art baseline models by an average of at least 6%. A case study on five primary fibrotic and closely related diseases reveals that ALIGATEHR-Gen effectively distinguishes patient subgroups based on clinical and genetic features. These findings illustrate the potential of ALIGATEHR-Gen to advance predictive and interpretable modeling in healthcare.

Introduction:
The modern healthcare system has been transformed by the proliferation of diverse data sources and multimodal patient information. Advances in digital health technologies and biomedical research have fostered a data-rich environment where electronic health records (EHR) and genetic data offer unprecedented opportunities to advance genomic medicine. EHRs serve as digitized repositories of patients' medical histories, including consultations, diagnoses, procedures, laboratory tests, and prescribed medications, while genetic data provides a comprehensive view of patients' genomic profiles, enabling the assessment of genetic contributions to disease risk.

The integration of multimodal data in EHR has facilitated the development of machine learning-based disease risk prediction models. In particular, deep learning architectures, such as recurrent neural networks (RNN), graph neural networks (GNN), and transformers, have demonstrated great potential in enhancing diagnostic predictions from EHR, suggesting the power of multimodal data integration in predictive healthcare analytics. Despite the considerable progress in predictive modeling across data modalities in EHR, the use of genetic information into clinical risk assessment remains limited.

Family health history is an important component to assess risk for common chronic diseases, because families share genetic variations, environmental exposures, common lifestyle, and health-related social factors. Unfortunately, family health history is often missing or inconsistently recorded in EHR databases. To address this limitation, the authors previously developed ALIGATEHR, which models the inferred family relations from de-identified demographic data in a graph attention network augmented with an attention-based medical ontology representation.

Large-scale biobanks, such as the UK Biobank (UKB), have become invaluable resources for investigating the role of genetics and its interplay with other factors in human health. By linking decades of EHR with genetic data, these biobanks provide a robust platform for assessing the risk of common diseases. The availability of genome-wide single nucleotide polymorphism (SNP) data enables the identification of familial relationships among participants.

ALIGATEHR-Gen (ALIgning Graph Attention neTworks for EHR using Genetic kinship) is a generic framework for learning patient representations to enhance disease risk prediction by integrating EHR and genetic data. ALIGATEHR-Gen learns patient representations in a graph attention network to account for family health history based on genetically inferred relationships. To further enhance the quality of the learned representations by accounting for misdiagnosis and disease dependencies, it additionally integrates a medical ontology of embedded diagnosis codes into the attention mechanism.

Core Contributions:
- An attention-based GNN framework, ALIGATEHR-Gen, for disease risk prediction. It is the first attention-based GNN model that integrates clinical features and genetic data for disease risk prediction, using genetically inferred first-degree relationships as graph edges.
- ALIGATEHR-Gen outperforms state-of-the-art baseline models across a broad spectrum of diseases in a large-scale real-world biobank dataset, the UKB. The ablation experiment confirms the robustness of the model.
- ALIGATEHR-Gen's ability to distinguish patient subgroups across five primary fibrotic or closely related diseases: coronary artery disease, chronic kidney disease, metabolic dysfunction-associated steatohepatitis, pulmonary fibrosis, and Crohn's disease.

Methods:
The ALIGATEHR-Gen framework consists of four main components: data processing, graph construction, an attention layer, and a prediction layer.

Data Processing:
Data from the UKB includes over 500,000 participants. Among them, 206,831 have both de-identified EHR, including both primary care and hospital inpatient data, and genetic data. Primary care data, released in 2019, capture participants' diagnosis history up to 2016 or 2017, while hospital inpatient data provide information on hospital admissions through 2022. Genetic data were released in 2017. Within this cohort, 63,693 participants have genetically inferred third-degree or closer relatives, and 22,364 have genetically inferred first-degree relatives.

Demographics of Experimental Population:
- UKB participants with first-degree relatives: 22,364 (60% female, 40% male, mean age at recruitment 56.9)
- CAD (coronary artery disease): 5,003 patients (33% female, 67% male, mean age at recruitment 61.3, mean age at first diagnosis 64.6)
- CKD (chronic kidney disease): 1,275 patients (51% female, 49% male, mean age at recruitment 62.1, mean age at first diagnosis 69.9)
- MASH (metabolic dysfunction-associated steatohepatitis): 220 patients (46% female, 54% male, mean age at recruitment 58.3, mean age at first diagnosis 64.6)
- PF (pulmonary fibrosis): 456 patients (48% female, 52% male, mean age at recruitment 56.8, mean age at first diagnosis 69.5)
- CD (Crohn's disease): 71 patients (53% female, 47% male, mean age at recruitment 56.3, mean age at first diagnosis 60.2)

Graph Construction:
ALIGATEHR-Gen constructs two graphs: a patient graph from genetic kinship and an ontology graph from medical ontology.

Patient Graph: To characterize familial relatedness within the UKB participants, kinship coefficients are estimated using genome-wide SNP data. The kinship coefficient represents the probability that a randomly selected allele from one individual is identical by descent (IBD) with an allele at the same locus from another individual. First-degree relatives are defined as pairs with a kinship coefficient >= 0.177. Based on these inferred relationships, a patient graph is then built by connecting first-degree relatives within the cohort.

Ontology Graph: In medical ontologies, the hierarchy of various medical concepts is represented through a parent-child relationship, with diagnosis codes serving as leaf nodes. The ontology graph is modeled as a directed acyclic graph (DAG), where parent nodes represent more general medical concepts and child nodes represent more specific subcategories.

Attention Layer:
Patient Graph Attention: In the patient graph, each node (patient) has an associated feature vector representing a patient's disease status. Between the nodes, an attention mechanism links a patient's representation to the clinical profiles of their relatives. The attention coefficients indicate the importance of each first-degree relative's disease features to the patient.

Ontology Graph Attention: In the ontology DAG, each node is assigned a basic embedding vector. The final representation of each diagnosis code is computed as a convex combination of the basic embeddings of itself and its ancestors, weighted by attention.

Prediction Layer:
The final representation for each patient at each visit is constructed by aggregating: (1) the patient's diagnosis status infused with information from first-degree relatives, (2) ontology representation from the ontology graph, and (3) patient demographic embedding including age and sex. The visit representation is then used as input to an LSTM-based prediction layer. The prediction task is to predict disease risk at the next visit given all previous visit history.

Experimental Setting:
The prediction of disease onset at a patient's next clinical visit is defined as a binary classification task. A total of 118 predictive models are developed, each corresponding to a unique disease code. These 118 diseases are common diseases from ten ICD coding categories: Diseases of the Blood and Blood-Forming Organs, Diseases of the Circulatory System, Diseases of the Digestive System, Diseases of the Genitourinary System, Diseases of the Musculoskeletal System and Connective Tissue, Diseases of the Nervous System and Sense Organs, Diseases of the Respiratory System, Endocrine Nutritional Metabolic Diseases and Immunity Disorders, Mental Disorders, and Neoplasms.

For all models, the dataset is randomly partitioned into training (70%), validation (10%), and testing (20%). Performance is evaluated using AUC, precision, recall, and F1 score with 3-fold cross validation.

Baseline Models:
- Group 1 (no attention mechanism): Logistic Regression (LR), XGBoost, and Recurrent Neural Networks (RNN).
- Group 2 (attention-based): GRAM and Dipole. GRAM leverages disease taxonomy and relies solely on patients' historical diagnosis codes as input.

Results:
Model Performance:
ALIGATEHR-Gen outperformed all baseline methods across 118 diseases:
- ALIGATEHR-Gen: AUC 0.76±0.06, Precision 0.78±0.06, Recall 0.71±0.06, F1 0.74±0.06
- GRAM (best baseline): AUC 0.72±0.06, Precision 0.74±0.05, Recall 0.68±0.06, F1 0.71±0.06
- Dipole: AUC 0.71±0.07, Precision 0.72±0.07, Recall 0.68±0.07, F1 0.70±0.07
- XGBoost: AUC 0.68±0.05, Precision 0.64±0.06, Recall 0.69±0.05, F1 0.66±0.05
- RNN: AUC 0.62±0.05, Precision 0.56±0.05, Recall 0.63±0.05, F1 0.59±0.05
- Logistic Regression: AUC 0.60±0.05, Precision 0.60±0.05, Recall 0.58±0.05, F1 0.59±0.05
This represents an improvement of 6% in average AUC compared to the best-performing baseline model.

Ablation Study:
- ALIGATEHR-Gen without patient graph: AUC 0.64±0.05
- ALIGATEHR-Gen with constant weights on patient graph edges: AUC 0.65±0.06
- Full ALIGATEHR-Gen: AUC 0.76±0.06
Removing the patient graph or assigning constant weights significantly affected performance, confirming the importance of incorporating first-degree relatives' health information with learned attention weights.

Case Study - Fibrotic and Closely Related Diseases:
A multi-class classifier was trained to distinguish among five primary fibrotic or closely related diseases: coronary artery disease (CAD), chronic kidney disease (CKD), metabolic-associated steatohepatitis (MASH), pulmonary fibrosis (PF), and Crohn's disease (CD).

A t-SNE plot of patient representations shows distinct separation between PF and CD patients from those with CAD, CKD, and MASH. CAD, CKD and MASH patients exhibit partial overlap, reflecting shared risk factors and underlying pathophysiological correlations.

Interpretability Analysis:
Across all five conditions, high-risk patients (predicted risk > 75%) consistently had a greater number of affected family members compared to moderate-risk (25-75%) and low-risk (< 25%) groups.

High-risk patients with CAD, CKD and MASH generally had higher BMI than those in moderate- and low-risk groups, even though BMI was not used in the prediction model. Patients with MASH had the highest BMI among all five diseases.

Discussion:
ALIGATEHR-Gen addresses the underutilization of family health history in disease risk prediction, where such information is often missing or incomplete. The improvement is driven by two factors:
1. Incorporating genetically inferred relationships captures latent genetic predispositions overlooked by EHR-centric models.
2. The attention mechanisms in both the patient graph and medical ontology graph enable prioritizing the most influential first-degree relatives and identifying clinically relevant features.

The case studies highlight the potential of multimodal learning for complex conditions with overlapping risk factors. The identified high-risk patients, characterized by familial predisposition and elevated BMI, demonstrate ALIGATEHR-Gen's capacity to capture interactions between genetic factors, demographics, and clinical features.

Limitations:
- The model focuses on genetically inferred first-degree relationships and may miss information from more distant relatives.
- Kinship coefficients do not fully capture shared environmental or lifestyle factors among relatives.
- Disease-specific polygenic risk scores (PRS) could further improve disease risk assessment if integrated.

Future Directions:
Future work will focus on validating the model in more diverse populations (e.g. All of Us Research Program) and addressing computational scalability, interoperability with EHR systems, and alignment with clinical workflows.

Conclusions:
ALIGATEHR-Gen is an attention-based graph neural network that integrates EHR, genetic data, and external medical ontology to improve disease risk prediction. It achieves superior predictive performance across 118 diseases, highlighting the importance of incorporating genetic information in patient representation learning.

Ethics:
This study uses data from the UK Biobank (project ID: 41910). UK Biobank received ethics approval from the North West Multi-centre Research Ethics Committee. All participants provided informed consent.

Funding:
Supported by NIH grant R01LM014087.
"""
