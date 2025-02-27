"""
Skin Disease Dataset Information

This file provides information about recommended datasets for training the skin disease classification model.

1. HAM10000 (Human Against Machine with 10000 training images) Dataset
   - Contains 10,000+ dermatoscopic images of pigmented skin lesions
   - 7 different classes of skin conditions
   - Available at: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
   - Paper: https://arxiv.org/abs/1803.10417

2. Dermnet Dataset
   - Contains 23,000+ images of skin diseases
   - 23 categories of skin diseases
   - Available at: https://www.kaggle.com/datasets/shubhamgoel27/dermnet

3. ISIC Archive (International Skin Imaging Collaboration)
   - Large collection of dermoscopic images
   - Regularly updated with new images
   - Available at: https://www.isic-archive.com/

4. SD-198 Dataset
   - Contains 6,584 clinical images of 198 skin diseases
   - Paper: https://arxiv.org/abs/1606.01258

How to prepare the dataset:

1. Download one of the datasets mentioned above
2. Organize the dataset into the following structure:
   
   dataset/
   ├── train/
   │   ├── melanoma/
   │   ├── eczema/
   │   ├── psoriasis/
   │   └── normal/
   └── validation/
       ├── melanoma/
       ├── eczema/
       ├── psoriasis/
       └── normal/

3. Preprocess the images:
   - Resize to 224x224 pixels
   - Normalize pixel values
   - Apply data augmentation for training set

4. Use the SkinDiseaseClassifier.train() method with the path to your dataset
"""

# Example code to download and prepare the HAM10000 dataset
def download_ham10000():
    """
    Downloads and prepares the HAM10000 dataset.
    This is a placeholder function - in a real implementation, you would:
    1. Download the dataset from the source
    2. Extract and organize the files
    3. Split into train/validation sets
    """
    print("To download and prepare the HAM10000 dataset:")
    print("1. Visit https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
    print("2. Download the dataset files")
    print("3. Extract the files to a directory")
    print("4. Use the following code to organize the dataset:")
    
    code_example = """
    import os
    import pandas as pd
    import shutil
    from sklearn.model_selection import train_test_split
    
    # Load metadata
    metadata = pd.read_csv('HAM10000_metadata.csv')
    
    # Create directories
    os.makedirs('dataset/train/melanoma', exist_ok=True)
    os.makedirs('dataset/train/eczema', exist_ok=True)
    os.makedirs('dataset/train/psoriasis', exist_ok=True)
    os.makedirs('dataset/train/normal', exist_ok=True)
    os.makedirs('dataset/validation/melanoma', exist_ok=True)
    os.makedirs('dataset/validation/eczema', exist_ok=True)
    os.makedirs('dataset/validation/psoriasis', exist_ok=True)
    os.makedirs('dataset/validation/normal', exist_ok=True)
    
    # Map HAM10000 diagnosis to our categories
    diagnosis_map = {
        'mel': 'melanoma',
        'nv': 'normal',
        'bcc': 'normal',
        'akiec': 'normal',
        'bkl': 'normal',
        'df': 'normal',
        'vasc': 'normal'
    }
    
    # Add custom mapping for eczema and psoriasis (not in HAM10000)
    # In a real scenario, you would need additional datasets for these conditions
    
    # Split data into train and validation
    train_df, val_df = train_test_split(metadata, test_size=0.2, random_state=42, stratify=metadata['dx'])
    
    # Copy images to appropriate directories
    def copy_images(df, train_or_val):
        for _, row in df.iterrows():
            image_id = row['image_id']
            diagnosis = diagnosis_map.get(row['dx'], 'normal')
            
            # Source path (adjust based on your download structure)
            src_path = f'HAM10000_images/{image_id}.jpg'
            
            # Destination path
            dst_path = f'dataset/{train_or_val}/{diagnosis}/{image_id}.jpg'
            
            # Copy the file
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
    
    # Copy images to train and validation directories
    copy_images(train_df, 'train')
    copy_images(val_df, 'validation')
    """
    
    print(code_example)

if __name__ == "__main__":
    download_ham10000()