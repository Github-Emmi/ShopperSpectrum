
# Shopper Spectrum Presentation Script

## Purpose

This script is designed for a professional screen recording and pitch video for
Shopper Spectrum: Customer Segmentation and Product Recommendations in
E-Commerce. It explains the business problem, project architecture, machine
learning workflow, codebase structure, documentation, tests, deployment path,
and real-world business value.

Use this as a guided narration while recording for YouTube, LinkedIn, Facebook,
portfolio pages, sponsor outreach, or investor-style project demonstrations.

Recommended video length: 15 to 25 minutes.

Important recording note: Do not open `.env`, `kaggle.json`, or any file that
may contain private credentials during the recording.

---

## 1. Opening Hook

### Show on screen

Open the project root folder:

```bash
cd "/Volumes/EmmiDev256G/Projects/Shopper Spectrum"
```

Show the repository tree briefly, then open `README.md`.

### Say

Welcome. In this video I am presenting Shopper Spectrum, a production-ready
machine learning and business intelligence project for e-commerce.

The goal of this project is simple but commercially important: help an online
retail business understand who its customers are, segment them based on buying
behavior, and recommend products that customers are likely to buy next.

This project combines data analytics, unsupervised machine learning,
collaborative filtering, model persistence, automated testing, and a Streamlit
web application into one end-to-end system.

The two main capabilities are:

1. Customer segmentation using RFM analysis and KMeans clustering. Combining RFM Analysis and K-Means Clustering creates a powerful, data-driven approach to customer segmentation. RFM evaluates Recency, Frequency, and Monetary value, while K-Means automatically groups customers into behavioral segments based on these variables, removing the need for manual scoring.
2. Product recommendations using item-based collaborative filtering and cosine similarity. An Item-Based Collaborative Filtering (IBCF) recommendation system suggests products by finding similarities between items based on historical user interactions (such as ratings, purchases, or clicks). Instead of matching similar users, it determines that Item A is similar to Item B because a large group of users interacted with both.Cosine Similarity is the mathematical metric used to calculate this likeness by measuring the angle between two multi-dimensional item vectors.

This is not only a notebook experiment. It includes documentation, tests,
trained model artifacts, deployment configuration, and an interactive app that
can be used by non-technical business users.

---

## 2. Business Problem

### Show on screen

Open `README.md`, then `documentations/SRS.md`.

### Say

E-commerce companies generate large volumes of transaction data every day.
However, raw transaction data does not automatically become business insight.

This project solves three practical business questions:

1. Which customers are most valuable?
2. Which customers are becoming inactive or at risk?
3. Which products should be recommended together to increase cross-sell and
   upsell revenue?

The customer segmentation module helps the business group customers into
actionable categories like High-Value, Regular, Occasional, and At-Risk.

The recommendation module helps the business power features like "customers
also bought", product bundles, product discovery pages, and personalized
marketing campaigns.

In a real-world e-commerce environment, these insights can support targeted
email campaigns, retention programs, personalized storefronts, inventory
planning, and revenue optimization.

---

## 3. High-Level Architecture

### Show on screen

Open `documentations/design.md`, then point to the architecture section.

### Say

The project follows a batch-first machine learning architecture with a
Streamlit serving layer.

At a high level, the data flow looks like this:

```text
online_retail.csv
  -> shopper_spectrum.ipynb
  -> data cleaning, EDA, RFM, KMeans, cosine similarity
  -> models/*.pkl and product_metadata.json
  -> app.py
  -> Streamlit dashboard for recommendations and segmentation
```

The notebook is the batch training layer. It reads the retail transaction
dataset, cleans the data, performs exploratory data analysis, engineers RFM
features, trains the clustering model, builds the product similarity matrix,
and exports trained artifacts.

The Streamlit app is the serving layer. It loads those trained artifacts and
uses them for real-time inference. A user can enter a product name and receive
similar products, or enter customer RFM values and receive a segment label.

This is a clean separation of responsibilities. Training happens offline in
the notebook. Inference happens online in the Streamlit app.

---

## 4. Architectural Design Patterns

### Show on screen

Open `documentations/design.md`, then `app.py`.

### Say

Several design patterns are used in the project.

The first pattern is layered architecture. The project is separated into a data
layer, model training layer, artifact layer, application layer, testing layer,
and documentation layer.

The second pattern is CQRS, or Command Query Responsibility Segregation. The
notebook acts as the command side because it transforms data, trains models,
and writes artifacts to the `models/` directory. The Streamlit app acts as the
query side because it reads already-trained artifacts and serves predictions.

The third pattern is singleton-style caching. In `app.py`, `load_models()` uses
`@st.cache_resource`, which means the KMeans model, scaler, label map,
similarity matrix, and product list are loaded once and reused. This avoids
expensive repeated disk reads.

The fourth pattern is helper or factory-style output generation.
`get_recommendations()` receives a product name and always returns a consistent
recommendation result: a list of product names and the matched product.

The fifth pattern is graceful fallback. If product visual metadata is missing,
the app generates deterministic fallback metadata so the UI still works. If
model files are missing in production, the app can attempt to download them
from Kaggle using environment credentials.

These patterns make the project more maintainable, more deployable, and easier
to explain to business stakeholders.

---

## 5. Repository Walkthrough

### Show on screen

Run:

```bash
find . -maxdepth 2 -type f -not -path './.git/*' -not -path './venv/*' | sort
```

### Say

The repository is compact and production-focused. It contains one main
Streamlit application, one enrichment utility, one notebook pipeline, generated
model and data assets, documentation, deployment configuration, and tests.

The main directories are:

`documentations/` contains the enterprise documentation package. This includes
the software requirements specification, system design, technical guide, QA
test plan, user guide, and operating procedures.

`models/` contains generated machine learning artifacts used by the
application. These are the trained KMeans model, scaler, label map, product
list, and similarity matrix.

`tests/` contains the pytest suite. It validates preprocessing, RFM
engineering, clustering, recommendation logic, model artifacts, non-functional
requirements, and end-to-end acceptance scenarios.

`venv/` is the local Python virtual environment.

`.git/` stores repository history and branch metadata.

`.pytest_cache/` and `__pycache__/` are runtime caches generated by pytest and
Python.

The source tree is intentionally simple, which makes it easier for a team to
review, maintain, deploy, and extend.

---

## 6. Core Files and What They Do

### Show on screen

Open each file briefly in this order: `app.py`, `shopper_spectrum.ipynb`,
`enrich_metadata.py`, `requirements.txt`, `render.yaml`, `pytest.ini`.

### Say

The most important file is `app.py`. This is the Streamlit production
application. It loads all model artifacts, provides the navigation sidebar,
renders the product recommendation page, renders the customer segmentation
page, and includes an About page for project explanation.

Inside `app.py`, `_download_from_kaggle()` supports production cold starts by
downloading model artifacts when they are missing.

`load_models()` loads the five required artifacts from the `models/` directory
and caches them with Streamlit.

`load_product_metadata()` loads the product visual metadata from
`product_metadata.json`.

`_render_aliexpress_card()` creates the e-commerce-style product cards used in
the recommendation UI.

`get_recommendations()` performs fuzzy matching and retrieves the top similar
products from the cosine similarity matrix.

`page_recommendations()` renders the recommendation interface.

`page_segmentation()` renders the RFM input form and predicts the customer's
segment.

`page_about()` explains the project inside the application.

`render_sidebar()` controls page navigation.

`main()` ties the whole Streamlit app together.

The second major file is `shopper_spectrum.ipynb`. This is the full machine
learning pipeline. It starts with data loading and understanding, then data
cleaning, exploratory data analysis, RFM feature engineering, KMeans clustering,
recommendation engine creation, model evaluation, and artifact export.

The third major file is `enrich_metadata.py`. This script builds
`product_metadata.json`. It reads products from `online_retail.csv`, enriches
them with product thumbnails, ratings, reviews, and prices using Serper or
SerpApi, and falls back to synthetic metadata when no API result is available.

`requirements.txt` defines the Python dependencies. It includes pandas, NumPy,
matplotlib, seaborn, Plotly, Streamlit, Kaggle, requests, notebook dependencies,
and a pinned scikit-learn version.

`render.yaml` defines deployment on Render. It installs dependencies and starts
the Streamlit app using the Render-provided port.

`pytest.ini` configures the test suite. It defines the test directory, naming
patterns, verbosity, and markers for unit, integration, system, acceptance,
slow, Kaggle, and non-functional tests.

`.gitignore` protects the repository by excluding virtual environments,
datasets, model artifacts, OS files, editor files, caches, and secrets.

---

## 7. Documentation Package

### Show on screen

Open the `documentations/` directory and each Markdown file briefly.

### Say

One of the strengths of this project is that it is documented like a real
software and machine learning delivery project.

`documentations/SRS.md` is the Software Requirements Specification. It defines
the project purpose, user classes, functional requirements, non-functional
requirements, constraints, and acceptance criteria.

`documentations/design.md` is the system architecture document. It explains
the batch layer, serving layer, data flow, RFM pipeline, clustering pipeline,
recommendation pipeline, design patterns, and technology decisions.

`documentations/technical.md` is the developer and ML engineer reference. It
explains the notebook architecture, key functions, Streamlit app flow, model
artifacts, dataset schema, dependencies, and common technical fixes.

`documentations/QA_test_plan.md` is the quality assurance plan. It maps the
project requirements to unit tests, integration tests, system tests,
acceptance tests, regression checks, and known risk areas.

`documentations/user_guide.md` is written for business users and analysts. It
explains how to run the app, use the recommendation page, use the segmentation
page, and interpret the results.

`documentations/SOP.md` provides standard operating procedures. It explains
initial setup, retraining, deployment, troubleshooting, notebook quality gates,
and maintenance schedules.

This documentation package makes the project easier to evaluate, hand over,
maintain, and present to stakeholders.

---

## 8. Dataset and Data Preparation

### Show on screen

Open `online_retail.csv` briefly, but do not scroll too long. Then show the
preprocessing tests or notebook cleaning section.

### Say

The dataset is an online retail transaction dataset. It includes fields like
invoice number, stock code, product description, quantity, invoice date, unit
price, customer ID, and country.

The local dataset contains over 500,000 transaction rows. The core cleaning
steps remove rows with missing customer IDs, remove cancelled invoices, remove
non-positive quantities and unit prices, remove duplicates, parse invoice
dates, and engineer `TotalPrice` as quantity multiplied by unit price.

After cleaning, the data becomes suitable for two machine learning tasks.

For customer segmentation, the pipeline calculates RFM features:

Recency means how recently the customer purchased.

Frequency means how many unique orders the customer placed.

Monetary means how much the customer spent.

For product recommendations, the pipeline builds a customer-product matrix and
uses product co-purchase behavior to calculate item-item cosine similarity.

---

## 9. Customer Segmentation Model

### Show on screen

Open the RFM and clustering sections in `shopper_spectrum.ipynb`, then open
the segmentation page in the app.

### Say

The segmentation model uses RFM analysis and KMeans clustering.

RFM is a proven marketing framework because it describes customer value using
three simple behavioral signals: how recently the customer purchased, how often
the customer purchases, and how much the customer spends.

The pipeline standardizes the RFM features using `StandardScaler`, then trains
a KMeans++ model with four clusters.

The clusters are mapped to business labels:

High-Value customers are recent, frequent, high-spending buyers.

Regular customers show steady purchasing behavior.

Occasional customers buy less frequently.

At-Risk customers have not purchased recently and need retention campaigns.

In the Streamlit app, the business user enters Recency, Frequency, and
Monetary values. The app scales the input using the same scaler fitted during
training, predicts the cluster, maps the cluster to a business label, and
displays a recommended business action.

This turns raw machine learning output into language that marketing,
operations, and management teams can act on.

---

## 10. Product Recommendation Engine

### Show on screen

Open `get_recommendations()` in `app.py`, then open the product recommendation
page in Streamlit.

### Say

The recommendation engine uses item-based collaborative filtering.

The notebook builds a customer-by-product matrix where rows represent
customers, columns represent products, and values represent purchase quantity.

The system then computes cosine similarity between products. Products that are
bought by similar customers, or frequently purchased together, become more
similar in the model.

In the app, the user can enter a product name. The system uses fuzzy matching
through Python's `difflib` module, so it can handle partial names, lowercase
input, or small spelling mistakes.

Once the product is matched, the app retrieves the most similar products from
`similarity_df.pkl`, excludes the original product, sorts by similarity, and
returns the top five recommendations.

This can be used in real business scenarios such as product bundles, "customers
also bought" widgets, merchandising decisions, and personalized product
discovery.

---

## 11. Product Visual Metadata

### Show on screen

Open `enrich_metadata.py` and `product_metadata.json`.

### Say

To make the recommendation page more realistic, the project includes product
visual metadata.

The enrichment script reads unique product names from the retail dataset and
attempts to fetch product thumbnails and visual metadata from image search
providers such as Serper and SerpApi.

If the API is unavailable, out of credits, or returns no useful result, the
script falls back to deterministic synthetic metadata. This ensures the UI does
not break when external services fail.

The Streamlit app uses this metadata to display e-commerce-style product cards
with images, prices, ratings, sold counts, and ranking badges.

This turns a plain recommendation list into a more polished product discovery
experience.

---

## 12. Model Artifacts

### Show on screen

Open the `models/` directory.

### Say

The `models/` directory contains the generated machine learning artifacts.

`kmeans_model.pkl` is the trained KMeans++ model used for customer
segmentation.

`scaler.pkl` is the fitted StandardScaler used to normalize RFM values before
prediction.

`label_map.pkl` maps cluster IDs to readable segment names like High-Value,
Regular, Occasional, and At-Risk.

`similarity_df.pkl` is the item-item cosine similarity matrix used by the
recommendation engine.

`product_list.pkl` is the list of valid product names used for fuzzy matching.

`.gitkeep` keeps the `models/` folder present in Git even though generated
model artifacts are normally ignored.

This artifact-based design allows training and serving to be separated. The
notebook generates the artifacts. The app loads and serves them.

---

## 13. Tests and Quality Assurance

### Show on screen

Open the `tests/` directory. Then show the recent test command.

```bash
./venv/bin/python -m pytest tests/
```

### Say

The project includes a comprehensive pytest suite.

The current test run passed 180 tests.

`tests/test_preprocessing.py` validates CSV loading, cleaning rules, duplicate
removal, date parsing, and total price calculation.

`tests/test_rfm_clustering.py` validates RFM feature engineering, scaling,
KMeans behavior, and label mapping.

`tests/test_recommendations.py` validates similarity matrix shape,
self-similarity, fuzzy matching, unknown-product handling, and top-N
recommendations.

`tests/test_model_artifacts.py` validates that the model artifacts exist, load
correctly, and have the expected types and behavior.

`tests/test_regression_live_models.py` validates the live model artifacts in
the local `models/` directory.

`tests/test_nonfunctional.py` checks performance, security, relative paths,
clustering quality, and recommendation scalability.

`tests/test_system_acceptance.py` validates end-to-end scenarios for both the
product recommendation flow and the customer segmentation flow.

This test suite shows that the project is not only visually presentable; it is
also verified against functional, system, acceptance, regression, and
non-functional requirements.

---

## 14. Deployment and Production Readiness

### Show on screen

Open `render.yaml`, `requirements.txt`, and the production section in
`README.md`.

### Say

The project is prepared for deployment using Render.

`render.yaml` defines a Python web service. The build command installs the
dependencies from `requirements.txt`. The start command runs the Streamlit app
on the Render-provided port.

For production cold starts, `app.py` can download missing model artifacts from
a Kaggle dataset using environment variables. This keeps large model artifacts
out of the Git repository while still allowing the app to restore them at
runtime.

The `.gitignore` file excludes local secrets, Kaggle credentials, datasets,
model artifacts, virtual environments, and cache files.

This is important because production-grade work is not just about building the
model. It is also about deployment, reproducibility, security, and operational
maintenance.

---

## 15. Live Demo Flow

### Show on screen

Start the Streamlit app:

```bash
streamlit run app.py
```

### Say

Now I will demonstrate the application.

The first page is the product recommendation engine. I can type a product name,
for example:

```text
WHITE HANGING HEART T-LIGHT HOLDER
```

When I click Get Recommendations, the app finds the closest product match,
retrieves the top five similar products from the cosine similarity matrix, and
displays them as product cards.

The app also includes a "Recommended for you" section. This section provides
dynamic product suggestions and updates based on search history.

Now I will move to the customer segmentation page.

Here I enter three business metrics:

Recency: how many days since the customer last purchased.

Frequency: how many orders the customer placed.

Monetary: how much the customer spent.

For example:

```text
Recency: 30
Frequency: 5
Monetary: 500
```

When I click Predict Customer Segment, the app transforms this input using the
trained scaler, predicts the cluster using the KMeans model, and displays the
customer segment with a recommended marketing action.

This is the point where machine learning becomes actionable business
intelligence.

---

## 16. Real-World Implementation Value

### Show on screen

Return to `README.md` or show the app result screens.

### Say

In a real e-commerce company, this project can support multiple business
functions.

Marketing teams can use the customer segments to target campaigns more
effectively.

Retention teams can focus on At-Risk customers with win-back offers.

Merchandising teams can use product similarity to build bundles and improve
product discovery.

Operations teams can use top-selling and related-product insights to support
inventory planning.

Product teams can embed the recommendation engine into an e-commerce storefront
as a "similar products" or "customers also bought" feature.

Management can use the system as a customer intelligence dashboard to
understand customer behavior at a glance.

The project is built with familiar, practical technologies: Python, pandas,
NumPy, scikit-learn, Streamlit, Plotly, pytest, Kaggle, and Render.

That makes it understandable, maintainable, and realistic for small and
medium-sized businesses.

---

## 17. Sponsor and Stakeholder Pitch

### Show on screen

Show the app and the documentation side by side if possible.

### Say

For sponsors, partners, and stakeholders, the value of Shopper Spectrum is that
it demonstrates a complete AI and data product lifecycle.

It starts from raw transaction data.

It cleans and prepares the data.

It performs exploratory analysis.

It builds customer segmentation with unsupervised learning.

It builds product recommendations with collaborative filtering.

It saves reusable model artifacts.

It serves those artifacts in an interactive web application.

It includes enterprise-style documentation.

It includes automated tests.

It includes deployment configuration.

And most importantly, it translates technical output into business actions.

This is the kind of project that can be extended into a full customer
intelligence platform for e-commerce businesses.

Future extensions could include automated retraining, cloud storage, MLflow
model tracking, a FastAPI backend, user authentication, dashboard analytics,
customer lifetime value prediction, churn prediction, real-time event tracking,
and integration with Shopify, WooCommerce, or enterprise CRM systems.

---

## 18. Files and Directories Reference

### Say

Here is a concise explanation of the files and directories in the codebase.

`app.py` is the main Streamlit application.

`shopper_spectrum.ipynb` is the data science and machine learning pipeline.

`enrich_metadata.py` builds product visual metadata for the recommendation UI.

`online_retail.csv` is the source transaction dataset.

`product_metadata.json` stores product visual metadata used by the app.

`requirements.txt` defines project dependencies.

`render.yaml` defines the Render deployment service.

`pytest.ini` configures pytest.

`.gitignore` protects generated files, datasets, model artifacts, caches, and
secrets from being committed.

`.python-version` defines the expected Python runtime.

`.env` is for local environment variables and secrets.

`kaggle.json` is for Kaggle credentials and must not be exposed.

`kernel-metadata.json` is notebook or Kaggle metadata.

`documentations/SRS.md` defines requirements.

`documentations/design.md` defines architecture and design decisions.

`documentations/technical.md` explains implementation details.

`documentations/QA_test_plan.md` defines the testing strategy.

`documentations/user_guide.md` explains usage for business users.

`documentations/SOP.md` explains setup, retraining, deployment, and
maintenance.

`models/kmeans_model.pkl` is the trained segmentation model.

`models/scaler.pkl` is the fitted scaler.

`models/label_map.pkl` maps cluster IDs to segment names.

`models/similarity_df.pkl` stores product similarity values.

`models/product_list.pkl` stores valid product names.

`models/.gitkeep` preserves the models directory in source control.

`tests/test_preprocessing.py` validates data cleaning.

`tests/test_rfm_clustering.py` validates segmentation logic.

`tests/test_recommendations.py` validates recommendation behavior.

`tests/test_model_artifacts.py` validates model artifact structure and loading.

`tests/test_regression_live_models.py` checks live model behavior.

`tests/test_nonfunctional.py` checks performance, security, portability, and
scalability.

`tests/test_system_acceptance.py` validates complete user-facing flows.

`tests/__init__.py` marks the test directory as a Python package.

`venv/` is the local Python environment.

`.git/` is Git repository metadata.

`.pytest_cache/` is pytest runtime cache.

`__pycache__/` contains Python bytecode cache files.

---

## 19. Known Engineering Notes

### Say

There are also a few engineering notes worth mentioning transparently.

The model artifacts were trained with scikit-learn 1.6.1, and the project pins
that version in `requirements.txt` to avoid compatibility issues.

The app uses pickle artifacts, which is practical for this scope, but in a
larger production platform I would consider model registries, stronger artifact
validation, and a managed model-serving API.

External product image enrichment depends on third-party API credits. The app
is designed to keep working even when those credits are unavailable by falling
back to synthetic metadata.

Secrets such as Kaggle credentials and API keys are handled through local files
or environment variables and should never be committed or shown during a public
recording.

These notes show that the project has been built with practical tradeoffs and
clear next steps for production hardening.

---

## 20. Closing Statement

### Show on screen

End on the Streamlit app home screen or the recommendation results page.

### Say

Shopper Spectrum demonstrates how raw e-commerce transaction data can be
converted into practical business intelligence.

It identifies customer segments, recommends similar products, supports
marketing decisions, and presents everything through a simple interactive web
application.

The project includes the full lifecycle: data preprocessing, EDA, machine
learning, recommendation logic, model persistence, application development,
documentation, testing, and deployment configuration.

This is the kind of solution that can help small and medium-sized e-commerce
businesses make better decisions, improve customer retention, increase
cross-selling opportunities, and turn historical transaction data into
actionable insight.

Thank you for watching this presentation of Shopper Spectrum.

---

## 21. Short Social Media Version

Use this version for a shorter LinkedIn, Facebook, or portfolio reel.

### Say

This is Shopper Spectrum, a customer segmentation and product recommendation
system for e-commerce.

It takes online retail transaction data, cleans it, performs exploratory data
analysis, builds RFM customer features, trains a KMeans clustering model, and
creates an item-based collaborative filtering recommendation engine.

The Streamlit app lets users enter a product name and receive similar product
recommendations, or enter Recency, Frequency, and Monetary values to predict a
customer segment.

The project includes a complete notebook pipeline, saved model artifacts,
product metadata enrichment, a production-style Streamlit dashboard,
documentation, deployment configuration, and 180 passing automated tests.

This project shows how machine learning can turn raw e-commerce transactions
into business actions like targeted marketing, retention campaigns, product
bundling, and personalized recommendations.

---

## 22. Recording Checklist

Before recording:

1. Make sure the virtual environment is active.
2. Do not open `.env` or `kaggle.json`.
3. Keep terminal output clean.
4. Confirm tests pass if you want to show the QA result.
5. Start Streamlit only after explaining the codebase structure.
6. Use one known product example for recommendations.
7. Use one RFM example for segmentation.
8. End with business value and next-step roadmap.

Useful commands:

```bash
source venv/bin/activate
python -m pytest tests/
streamlit run app.py
```

Suggested demo inputs:

```text
Product: WHITE HANGING HEART T-LIGHT HOLDER
Recency: 30
Frequency: 5
Monetary: 500
```


