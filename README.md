# Image Similarity Search

## Using FAISS index to reverse search images

### Setup
-> To run the repo create a python virtual environment and install the packanges present in req.txt. using the command ```python3 -m pip install -r req.txt```.
-> Create folders /index and /images, to save the training images and trained indexes.
-> If the dataset is from s3 bucket the code can be used as is. If not source images from local directory using PIL library.

### Working
-> We encode the images using ResNet-18 trained on ImageNet1K, to encode the images.
-> We then use this encodings to train the FAISS index. For more info on usage of FAISS refer https://www.pinecone.io/learn/series/faiss/faiss-tutorial/.
-> Using this index we perform similarity search that can search more than a million records in fewer milli seconds.
