import os
import pandas as pd
import numpy as np
import cv2
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct

# Initialize Spark Session
spark = SparkSession.builder.appName("SkinDiseasePreprocessing").getOrCreate()

# Define data paths
DATA_DIR = "C:/Users/HP/OneDrive/Desktop/DermDetect/data"
IMG_DIR_1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
IMG_DIR_2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

# Load Metadata
df = spark.read.csv(METADATA_PATH, header=True, inferSchema=True)
df.show(5)

# Check for missing values
df.select([countDistinct(col(c)).alias(c) for c in df.columns]).show()

# Drop missing values
df = df.na.drop()

# Remove duplicate records
df = df.dropDuplicates()

# Save cleaned data
df.toPandas().to_csv(os.path.join(DATA_DIR, "HAM10000_metadata_cleaned.csv"), index=False)
print("✅ Data Preprocessing Complete!")
