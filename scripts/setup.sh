# copy client into backend directory
cp -r client/ backend/static
# copy model into /model directory from s3
curl -o model/checkpoint.pth https://built-model-repository.s3.us-west-2.amazonaws.com/cat_dog_classifier/1.2/checkpoint.pth
# install dependencies for flask app and ML framework
pip install -r backend/requirements.txt
pip install -r model/requirements.txt