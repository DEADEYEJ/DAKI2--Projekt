predict_all_models bundle

Indhold:
- predict_all_models.py
- Regex_importer.py
- crf_pii_model_v5.pkl
- crf5_pipeline\
- svm\SVM_supercomputer\pii_classifier.py
- svm\SVM_supercomputer\svm_hybrid_hpc_20260511_132650\svm_pii_classifier.pkl

Kør fra denne mappe:

python predict_all_models.py

eller med tekst direkte:

python predict_all_models.py "test@example.com"

eller fuld JSON:

python predict_all_models.py --json "test@example.com"

Bemærk:
- SVM-modellen kan stadig give sklearn versionsadvarsler ved load.
