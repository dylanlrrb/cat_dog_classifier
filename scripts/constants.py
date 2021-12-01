WORKING_DIR = 'model'
S3_BUCKET = 'built-model-repository'
S3_BUCKET_DIR = 'cat_dog_classifier'
PYTHON_VERSION = 'python3.7'
DOCKER_IMAGE_NAME = 'cat_dog'
VOLUME_MAPPINGS = [
  '/model:/src',
  '/container_cache/torch:/root/.cache/torch/checkpoints'
]
BUILD_PRODUCTS = ['checkpoint.pt', 'index.html']
CHECKPOINT_VERSION = "1.1"
NOTEBOOK_NAME = 'Cat_Dog_Classifier.ipynb'