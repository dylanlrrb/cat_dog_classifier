# copy client into backend directory
cp -r client/ backend/static
# copy model into /model directory from s3
curl -o model/checkpoint.pt https://built-model-repository.s3.us-west-2.amazonaws.com/cat_dog_classifier/1.1/checkpoint.pt
# install dependencies for flask app and ML framework
pip install -r backend/requirements.txt
pip install -r model/requirements.txt