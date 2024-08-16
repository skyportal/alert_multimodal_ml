# ML_skyportal

- **`kowalski.py`**: Loads the Kowalski dataset.
- **`processor/`**:
  - **`photometry_processor.py`**: Preprocesses photometry data.
  - **`alert_processor.py`**: Handles alerts, including images and metadata.
  - **`data_processor.py`**: General data preprocessing tasks.
- **`supernova_dataset.py`**: Aggregates functions from processors for comprehensive data preparation.
- **`data_generator.py`**: Inherits from Keras, manages data during model training.

- **Data Loading and Preprocessing**:
  - Use `kowalski.py` to load data.
  - Utilize `processor` classes for specific preprocessing needs.
  - `supernova_dataset.py` combines all preprocessing steps.

- **Data Generation for Training**:
  - `data_generator.py` provides batch data generation for Keras training.

- **Detailed Function Usage**:
  - Refer to the `Show_data` notebook for detailed instructions on each function, from loading data to training a model.
