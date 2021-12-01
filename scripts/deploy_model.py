import boto3
import constants
from  os import remove
import sys
import traceback
from datetime import datetime
from subprocess import check_output
from get_dataset import verify_and_download

# This script assumes you have the package 'boto3' installed and accessible from here
s3_client = boto3.client('s3')

def try_to_remove(file):
  try:
    remove(f'{constants.WORKING_DIR}/{file}')
  except FileNotFoundError:
    print(f'{constants.WORKING_DIR}/{file} already removed or does not exist')


def build():
  # fetch dataset at this commit if not cached already
  verify_and_download()
  for file in constants.BUILD_PRODUCTS:
    try_to_remove(file)
  out = check_output(['bash', 'scripts/run_container.sh', constants.PYTHON_VERSION, constants.NOTEBOOK_NAME, constants.DOCKER_IMAGE_NAME, *constants.VOLUME_MAPPINGS])
  print(out.decode("utf-8"))

def push_tag (version):
  for file in constants.BUILD_PRODUCTS:
    extra_args = {
        'ACL': 'public-read',
      }
    if file == 'index.html':
      extra_args = {
        'ACL': 'public-read',
        'ContentType': 'text/html',
        'ContentDisposition': 'inline'
      }

    s3_client.upload_file(
      f'{constants.WORKING_DIR}/{file}',
      constants.S3_BUCKET,
      f'{constants.S3_BUCKET_DIR}/{version}/{file}',
      ExtraArgs=extra_args
    )
    
    try_to_remove(file)


# ------------------------------------------------

if __name__ == "__main__":
  dt_string = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
  print('<<<<<<<<< START OF EXECUTION >>>>>>>>>  ', dt_string)	

  try:    
    build()

    push_tag(constants.CHECKPOINT_VERSION)

    print('----------------------')
  except Exception as e:      
    with open('index.html', 'w') as file:
      file.write(str(e))
      traceback.print_tb(sys.exc_info()[-1], limit=None, file=file)
    s3_client.upload_file(
      'index.html',
      constants.S3_BUCKET,
      f'{constants.S3_BUCKET_DIR}/{constants.CHECKPOINT_VERSION}/index.html',
      ExtraArgs={
        'ACL': 'public-read',
        'ContentType': 'text/html',
        'ContentDisposition': 'inline'
      }
    )

  print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< END OF EXECUTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n\n\n\n')
