Sentiment-Driven Analysis of Customer Complaints for Adaptive Product Recommendations Using Large Language Models
Project overview
This repository contains the Jupyter notebook sentiment-analysis_nb.ipynb, developed for the thesis “Sentiment-Driven Analysis of Customer Complaints for Adaptive Product Recommendations Using Large Language Models.”

The notebook explores how customer complaints can be analysed and used as additional signals in product recommendation. It combines review ratings and text with sentiment, emotion, complaint-aspect and intensity features. In the later multimodal stages, it also processes customer-uploaded review images and derives visual representations with CLIP before comparing recommendation/ranking approaches.

The notebook is designed primarily for a Kaggle Notebook environment. It uses Kaggle-specific input and output paths, and it expects the necessary datasets to be attached to the notebook.
Main workflow
1. Review cleaning and complaint filtering – standardises review fields, removes missing or short review text, and selects low-rated reviews for complaint analysis.
2. Sentiment analysis – uses the Hugging Face model distilbert-base-uncased-finetuned-sst-2-english to produce sentiment labels and confidence scores.
3. Emotion analysis – uses j-hartmann/emotion-english-distilroberta-base to estimate emotion labels and scores.
4. Complaint aspects and intensity – identifies aspects such as quality, delivery, price, service, packaging and usability using keyword rules, then calculates a heuristic intensity score.
5. User and interaction features – builds complaint profiles and user–product rating/interactions for recommendation experiments.
6. Recommendation experiments – compares an SVD collaborative-filtering baseline with feature-augmented approaches and XGBoost-based prediction/reranking models. Ranking evaluation includes Precision@K, Recall@K and NDCG@K.
7. Review-image processing – extracts image references from review data, downloads and validates available images, and uses openai/clip-vit-base-patch32 to generate visual features.
8. Multimodal fusion and reporting – combines text and visual features for text-only, image-only and multimodal experiments, then creates analysis tables and figures for the thesis.
Some notebook cells are exploratory or represent earlier versions of the pipeline. The later stages rely on intermediate files or data that may already have been created in a previous Kaggle run. For that reason, this notebook should not be assumed to run from top to bottom in a fresh environment until all required inputs and paths have been checked.
Dataset sources
The notebook does not include the raw datasets. Obtain/access the source data separately and comply with the relevant dataset terms.
- Amazon-combined dataset package on Hugging Face: https://huggingface.co/datasets/randomath/Amazon-combined
- Underlying Amazon Reviews 2023 dataset page: https://mcauleylab.ucsd.edu/public_datasets/amazon_demo/
The Hugging Face package is the dataset package referenced for accessing product metadata and review records, including customer-posted image references. These links describe related sources; they should not be presented as two independent datasets used in the experiment.
Files expected by the notebook
The notebook contains hard-coded Kaggle paths. Depending on which stages you run, attach the corresponding data files using Kaggle's Add Input feature, or update the paths in the notebook to match your environment.
File or dataset directory	Purpose in the notebook
/kaggle/input/datasets/vrindabhatia108/reviews-dataset/Reviews.csv	Input for the initial exploratory complaint-analysis and baseline stages.
/kaggle/input/datasets/vrindabhatia108/combined-json/combined.json	JSON source for extracting review records and image references in the later pipeline.
/kaggle/input/datasets/vrindabhatia108/complaint-reviews-annotated/complaint_reviews_annotated.csv	Annotated complaint records used by the earlier model experiments.
/kaggle/input/datasets/vrindabhatia108/user-complaint-profiles/user_complaint_profiles.csv	Aggregated complaint profiles for users.
/kaggle/input/datasets/vrindabhatia108/user-item-ratings/user_item_ratings.csv	User–product ratings/interactions for recommendation experiments.
/kaggle/input/datasets/vrindabhatia108/thesis-data/final_multimodal_dataset.csv	Prepared interaction-level dataset used by the thesis analysis/results stages.
/kaggle/input/datasets/vrindabhatia108/thesis-data/review_images_downloaded.csv	Downloaded/validated review-image records used in the Chapter 4 analysis stage.


Some of the CSV files above are intermediate or prepared datasets, rather than separate raw data sources. If they are not available in your Kaggle account, run the relevant earlier processing cells to create them or update the paths to point to your own copies.
Environment and dependencies
A GPU-enabled Kaggle Notebook is recommended for transformer inference and CLIP image embedding. The notebook includes installation cells for some dependencies. For a fresh Python environment, the core packages used include:
pip install numpy pandas scipy scikit-learn matplotlib xgboost transformers torch datasets tqdm ijson requests pillow accelerate kagglehub
The pretrained Hugging Face model weights are downloaded when their respective model-loading cells run, so internet access may be required. GPU-dependent cells may need adjustment if you run the notebook on CPU; for example, some pipeline cells set device=0 for a CUDA GPU.
How to run
1. Open sentiment-analysis_nb.ipynb in Kaggle Notebooks.
2. Attach the required Kaggle datasets via Add Input. Verify that the file names and directory paths match the paths listed above.
3. Select a GPU accelerator for the transformer and CLIP stages, where available.
4. Run the notebook sections in order, checking that each stage's expected input CSV has been created before moving to the next stage. The notebook contains multiple historical/exploratory sections, so skip an earlier section only when its required intermediate files already exist and are compatible with the next section.
5. If running outside Kaggle, replace /kaggle/input/... and /kaggle/working/... paths with paths appropriate to your local environment.
6. After execution, inspect the generated CSV tables and PNG figures and confirm that the outputs correspond to the experiment section being reported.
Outputs
Depending on the stages executed, the notebook can create intermediate data files and plots including:
- complaint_reviews_annotated.csv
- user_complaint_profiles.csv
- user_item_ratings.csv
- reviews_with_images.csv, all_reviews.csv and model_reviews.csv
- complaints_sentiment.csv, complaints_text_features.csv and final_complaint_text_features.csv
- final_interaction_features.csv
- review_images.csv, review_images_downloaded.csv and review_visual_features.csv
- multimodal_interaction_features.csv and final_multimodal_dataset.csv
- Recommendation-result CSV files, Chapter 4 analysis tables/figures and Chapter 5 results tables/figures
In Kaggle, outputs are generally written to /kaggle/working/. Download the required outputs from the Kaggle session if you need to archive or submit them.
Methodological notes
- Complaint intensity is calculated with a heuristic based on the predicted emotion and its score; it is not a separately trained intensity model.
- Complaint aspect extraction in this notebook is keyword-based rather than a fine-tuned aspect-based sentiment-analysis model.
- CLIP is used to derive image representations and similarity-based visual signals. It should not be described as a dedicated product-defect detection model.
- Results depend on the exact dataset version, preprocessing, available images, random seeds, model/library versions and the execution path through the notebook.
Suggested citation / acknowledgement
If you use the notebook or reproduce the experiments, cite the thesis by its approved title and acknowledge the Amazon Reviews 2023 data source and the Hugging Face package through which the review data was accessed. Follow the dataset providers' terms when reusing or redistributing data. Do not upload raw review data or customer images into a public code repository unless the dataset terms explicitly permit it.
